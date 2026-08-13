package genaiengine

import (
	"context"
	"io"
	"iter"
	"strconv"

	"google.golang.org/genai"

	"github.com/lf4096/koine"
)

// syntheticIDPrefix marks tool-call ids the engine minted because the model
// returned a FunctionCall without one; they are stripped on resend.
const syntheticIDPrefix = "koine:"

// LanguageOptions carries request parameters specific to genai-based providers. Pass
// it via LanguageRequest.ProviderOptions keyed by the provider name.
type LanguageOptions struct {
	SafetySettings []*genai.SafetySetting
	// ThinkingConfig replaces the mapping from the normalized Thinking.
	ThinkingConfig *genai.ThinkingConfig
}

// Stream performs one streaming generateContent call.
func (e *Engine) Stream(ctx context.Context, req *koine.LanguageRequest) (*koine.LanguageStream, error) {
	contents, config, err := e.encodeRequest(req)
	if err != nil {
		return nil, e.wrapErr(err)
	}
	pull, stop := iter.Pull2(e.Client.Models.GenerateContentStream(ctx, e.Model, contents, config))

	acc := &accumulator{name: e.Name}
	var pending []koine.Event
	blockIndex := -1
	blockKind := ""
	stopSent := false

	next := func() (koine.Event, error) {
		for {
			if len(pending) > 0 {
				ev := pending[0]
				pending = pending[1:]
				return ev, nil
			}
			resp, err, ok := pull()
			if err != nil {
				return koine.Event{}, e.wrapErr(err)
			}
			if !ok {
				if !stopSent {
					stopSent = true
					usage := acc.usage
					pending = append(pending,
						koine.Event{Type: koine.EventUsage, Usage: &usage},
						koine.Event{Type: koine.EventStop, StopReason: acc.stopReason()},
					)
					continue
				}
				return koine.Event{}, io.EOF
			}
			acc.addResponse(resp)
			if len(resp.Candidates) == 0 || resp.Candidates[0].Content == nil {
				continue
			}
			for _, part := range resp.Candidates[0].Content.Parts {
				switch {
				case part.FunctionCall != nil:
					blockKind = "tool"
					blockIndex++
					id := part.FunctionCall.ID
					if id == "" {
						id = syntheticIDPrefix + strconv.Itoa(blockIndex)
					}
					input := acc.addFunctionCall(part, id)
					name := part.FunctionCall.Name
					pending = append(pending,
						koine.Event{Type: koine.EventToolCallStart, Index: blockIndex, ToolCall: &koine.ToolCallEvent{ID: id, Name: name}},
						koine.Event{Type: koine.EventToolCallEnd, Index: blockIndex, ToolCall: &koine.ToolCallEvent{ID: id, Name: name, Input: input}},
					)
				case part.Thought && part.Text != "":
					if blockKind != "thinking" {
						blockKind = "thinking"
						blockIndex++
					}
					acc.addThinking(part)
					pending = append(pending, koine.Event{Type: koine.EventThinkingDelta, Index: blockIndex, Text: part.Text})
				case part.Text != "":
					if blockKind != "text" {
						blockKind = "text"
						blockIndex++
					}
					acc.addText(part)
					pending = append(pending, koine.Event{Type: koine.EventTextDelta, Index: blockIndex, Text: part.Text})
				default:
					acc.addOther(part)
				}
			}
		}
	}
	final := func() (*koine.LanguageResponse, error) {
		return acc.response(), nil
	}
	closeFn := func() error {
		stop()
		return nil
	}
	return koine.NewLanguageStream(next, final, closeFn), nil
}
