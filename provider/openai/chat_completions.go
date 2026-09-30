package openai

import (
	"context"
	"encoding/json"
	"io"
	"strings"

	"github.com/openai/openai-go/v3"

	"github.com/lf4096/koine"
)

// ChatCompletionsModel speaks the OpenAI Chat Completions protocol for one
// model, for OpenAI-compatible endpoints that do not serve the Responses API.
type ChatCompletionsModel struct {
	model  string
	client openai.Client
}

var _ koine.LanguageModel = (*ChatCompletionsModel)(nil)

// ChatCompletionsModel builds the chat model on Chat Completions.
func (p *Provider) ChatCompletionsModel(model string) *ChatCompletionsModel {
	return &ChatCompletionsModel{model: model, client: p.client}
}

func (m *ChatCompletionsModel) Model() string { return m.model }

func (m *ChatCompletionsModel) Provider() string { return Name }

func (m *ChatCompletionsModel) Capabilities() koine.LanguageCapabilities {
	return koine.LanguageCapabilities{StopSequences: true, Thinking: true, ParallelToolCalls: true, Images: true, StructuredOutput: true}
}

// toolState assembles one streamed tool call; Chat Completions deltas carry
// no block structure, so the provider synthesizes koine block indices itself.
type toolState struct {
	id    string
	name  string
	args  strings.Builder
	index int
	ended bool
}

// Complete performs one Chat Completions call and returns the final response.
func (m *ChatCompletionsModel) Complete(ctx context.Context, req *koine.LanguageRequest) (*koine.LanguageResponse, error) {
	s, err := m.Stream(ctx, req)
	if err != nil {
		return nil, err
	}
	return s.Collect()
}

// Stream performs one streaming Chat Completions call.
func (m *ChatCompletionsModel) Stream(ctx context.Context, req *koine.LanguageRequest) (*koine.LanguageStream, error) {
	params, reqOpts, err := encodeChatRequest(m.model, req)
	if err != nil {
		return nil, wrapErr(err)
	}
	sse := m.client.Chat.Completions.NewStreaming(ctx, params, reqOpts...)

	var (
		acc        openai.ChatCompletionAccumulator
		pending    []koine.Event
		tools      = map[int64]*toolState{}
		toolOrder  []int64
		reasoning  strings.Builder
		field      string
		usage      koine.Usage
		finish     string
		blockKind  string
		blockIndex = -1
		stopSent   bool
	)

	flushToolEnds := func() {
		for _, i := range toolOrder {
			st := tools[i]
			if st.ended {
				continue
			}
			st.ended = true
			pending = append(pending, koine.Event{
				Type:     koine.EventToolCallEnd,
				Index:    st.index,
				ToolCall: &koine.ToolCallEvent{ID: st.id, Name: st.name, Input: json.RawMessage(callArguments(st.args.String()))},
			})
		}
	}

	next := func() (koine.Event, error) {
		for {
			if len(pending) > 0 {
				ev := pending[0]
				pending = pending[1:]
				return ev, nil
			}
			if !sse.Next() {
				if err := sse.Err(); err != nil {
					return koine.Event{}, wrapErr(err)
				}
				if !stopSent {
					stopSent = true
					flushToolEnds()
					pending = append(pending, koine.Event{Type: koine.EventStop, StopReason: chatStopReason(finish)})
					continue
				}
				return koine.Event{}, io.EOF
			}
			chunk := sse.Current()
			acc.AddChunk(chunk)
			if u, ok := decodeChatUsage(chunk.Usage); ok {
				usage = u
				pending = append(pending, koine.Event{Type: koine.EventUsage, Usage: &u})
			}
			if len(chunk.Choices) == 0 {
				continue
			}
			choice := chunk.Choices[0]
			delta := choice.Delta
			if name, text, ok := reasoningDelta(delta); ok {
				reasoning.WriteString(text)
				field = name
				if blockKind != "thinking" {
					blockKind = "thinking"
					blockIndex++
				}
				pending = append(pending, koine.Event{Type: koine.EventThinkingDelta, Index: blockIndex, Text: text})
			}
			if delta.Content != "" {
				if blockKind != "text" {
					blockKind = "text"
					blockIndex++
				}
				pending = append(pending, koine.Event{Type: koine.EventTextDelta, Index: blockIndex, Text: delta.Content})
			}
			for _, tc := range delta.ToolCalls {
				st := tools[tc.Index]
				if st == nil {
					blockKind = "tool"
					blockIndex++
					st = &toolState{id: tc.ID, name: tc.Function.Name, index: blockIndex}
					tools[tc.Index] = st
					toolOrder = append(toolOrder, tc.Index)
					pending = append(pending, koine.Event{
						Type:     koine.EventToolCallStart,
						Index:    st.index,
						ToolCall: &koine.ToolCallEvent{ID: st.id, Name: st.name},
					})
				} else {
					if st.id == "" {
						st.id = tc.ID
					}
					if st.name == "" {
						st.name = tc.Function.Name
					}
				}
				if tc.Function.Arguments != "" {
					st.args.WriteString(tc.Function.Arguments)
					pending = append(pending, koine.Event{
						Type:     koine.EventToolCallDelta,
						Index:    st.index,
						ToolCall: &koine.ToolCallEvent{ID: st.id, Name: st.name, InputDelta: tc.Function.Arguments},
					})
				}
			}
			if choice.FinishReason != "" {
				finish = choice.FinishReason
				flushToolEnds()
			}
		}
	}
	final := func() (*koine.LanguageResponse, error) {
		return decodeChatResponse(&acc, field, reasoning.String(), tools, toolOrder, finish, usage), nil
	}
	return koine.NewLanguageStream(next, final, sse.Close), nil
}

// reasoningFields are the non-standard fields OpenAI-compatible vendors
// stream reasoning in; each vendor uses one and replay must send it back
// under the same name.
var reasoningFields = []string{"reasoning_content", "reasoning"}

func reasoningDelta(delta openai.ChatCompletionChunkChoiceDelta) (name, text string, ok bool) {
	for _, candidate := range reasoningFields {
		// Extra fields carry status "invalid" in respjson (no schema to
		// validate against), so presence is tested via Raw, not Valid.
		field, found := delta.JSON.ExtraFields[candidate]
		if !found || field.Raw() == "" || field.Raw() == "null" {
			continue
		}
		if json.Unmarshal([]byte(field.Raw()), &text) == nil && text != "" {
			return candidate, text, true
		}
	}
	return "", "", false
}
