package anthropic

import (
	"context"
	"io"

	"github.com/anthropics/anthropic-sdk-go"

	"github.com/lf4096/koine"
)

// LanguageOptions carries Anthropic-specific request parameters. Pass it via
// LanguageRequest.ProviderOptions[anthropic.Name].
type LanguageOptions struct {
	TopK *int
}

// LanguageModel speaks the Anthropic Messages protocol for one model.
type LanguageModel struct {
	model  string
	client anthropic.Client
}

var _ koine.LanguageModel = (*LanguageModel)(nil)

// LanguageModel builds the chat model.
func (p *Provider) LanguageModel(model string) *LanguageModel {
	return &LanguageModel{model: model, client: p.client}
}

func (m *LanguageModel) Model() string { return m.model }

func (m *LanguageModel) Provider() string { return Name }

func (m *LanguageModel) Capabilities() koine.LanguageCapabilities {
	return koine.LanguageCapabilities{Thinking: true, CacheControl: true, ParallelToolCalls: true, Images: true, StructuredOutput: true}
}

// Complete performs one Messages call and returns the final response.
func (m *LanguageModel) Complete(ctx context.Context, req *koine.LanguageRequest) (*koine.LanguageResponse, error) {
	s, err := m.Stream(ctx, req)
	if err != nil {
		return nil, err
	}
	return s.Collect()
}

// Stream performs one streaming Messages call.
func (m *LanguageModel) Stream(ctx context.Context, req *koine.LanguageRequest) (*koine.LanguageStream, error) {
	params, err := encodeRequest(m.model, req)
	if err != nil {
		return nil, wrapErr(err)
	}
	sse := m.client.Messages.NewStreaming(ctx, params)

	var acc anthropic.Message
	// Some SDK events expand to two koine events (message_delta carries both
	// usage and stop); the queue holds the overflow.
	var pending []koine.Event
	tools := map[int64]*koine.ToolCallEvent{}

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
				return koine.Event{}, io.EOF
			}
			event := sse.Current()
			if err := acc.Accumulate(event); err != nil {
				return koine.Event{}, wrapErr(err)
			}
			switch ev := event.AsAny().(type) {
			case anthropic.ContentBlockStartEvent:
				if ev.ContentBlock.Type == "tool_use" {
					tc := &koine.ToolCallEvent{ID: ev.ContentBlock.ID, Name: ev.ContentBlock.Name}
					tools[ev.Index] = tc
					return koine.Event{Type: koine.EventToolCallStart, Index: int(ev.Index), ToolCall: tc}, nil
				}
			case anthropic.ContentBlockDeltaEvent:
				switch delta := ev.Delta.AsAny().(type) {
				case anthropic.TextDelta:
					return koine.Event{Type: koine.EventTextDelta, Index: int(ev.Index), Text: delta.Text}, nil
				case anthropic.ThinkingDelta:
					return koine.Event{Type: koine.EventThinkingDelta, Index: int(ev.Index), Text: delta.Thinking}, nil
				case anthropic.InputJSONDelta:
					tc := tools[ev.Index]
					if tc == nil {
						continue
					}
					return koine.Event{
						Type:  koine.EventToolCallDelta,
						Index: int(ev.Index),
						ToolCall: &koine.ToolCallEvent{
							ID:         tc.ID,
							Name:       tc.Name,
							InputDelta: delta.PartialJSON,
						},
					}, nil
				}
			case anthropic.ContentBlockStopEvent:
				tc := tools[ev.Index]
				if tc == nil {
					continue
				}
				input := acc.Content[ev.Index].Input
				if len(input) == 0 {
					input = []byte("{}")
				}
				return koine.Event{
					Type:     koine.EventToolCallEnd,
					Index:    int(ev.Index),
					ToolCall: &koine.ToolCallEvent{ID: tc.ID, Name: tc.Name, Input: input},
				}, nil
			case anthropic.MessageDeltaEvent:
				usage := koine.Usage{
					InputTokens:      int(ev.Usage.InputTokens),
					OutputTokens:     int(ev.Usage.OutputTokens),
					CacheReadTokens:  int(ev.Usage.CacheReadInputTokens),
					CacheWriteTokens: int(ev.Usage.CacheCreationInputTokens),
					ReasoningTokens:  int(ev.Usage.OutputTokensDetails.ThinkingTokens),
				}
				pending = append(pending, koine.Event{Type: koine.EventStop, StopReason: mapStopReason(ev.Delta.StopReason)})
				return koine.Event{Type: koine.EventUsage, Usage: &usage}, nil
			}
		}
	}
	final := func() (*koine.LanguageResponse, error) {
		return decodeResponse(&acc), nil
	}
	return koine.NewLanguageStream(next, final, sse.Close), nil
}
