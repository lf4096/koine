package openai

import (
	"cmp"
	"context"
	"io"

	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/responses"

	"github.com/lf4096/koine"
)

// LanguageOptions carries OpenAI request parameters for both LanguageModel
// and ChatCompletionsModel. Pass it via
// LanguageRequest.ProviderOptions[openai.Name].
type LanguageOptions struct {
	// LegacyMaxTokens sends max_tokens instead of max_completion_tokens, for
	// compatible endpoints that predate the rename. Chat Completions only.
	LegacyMaxTokens bool
	// NoStreamUsage omits stream_options.include_usage, for compatible
	// endpoints that reject it. Chat Completions only.
	NoStreamUsage bool
	// NoReasoningReplay drops thinking on replay instead of sending it back
	// under its vendor field, for compatible endpoints that reject the field
	// in input messages. Chat Completions only.
	NoReasoningReplay bool
	// ExtraBody merges additional top-level fields into the request body
	// (e.g. vendor thinking switches like enable_thinking).
	ExtraBody map[string]any
}

// LanguageModel speaks the OpenAI Responses protocol for one model.
type LanguageModel struct {
	model  string
	client openai.Client
}

var _ koine.LanguageModel = (*LanguageModel)(nil)

// LanguageModel builds the chat model on the Responses API.
func (p *Provider) LanguageModel(model string) *LanguageModel {
	return &LanguageModel{model: model, client: p.client}
}

func (m *LanguageModel) Model() string { return m.model }

func (m *LanguageModel) Provider() string { return Name }

func (m *LanguageModel) Capabilities() koine.LanguageCapabilities {
	// Prompt caching is automatic on OpenAI; there is no cache control to
	// express, so CacheControl stays false.
	return koine.LanguageCapabilities{Thinking: true, ParallelToolCalls: true, Images: true, StructuredOutput: true}
}

// Complete performs one Responses call and returns the final response.
func (m *LanguageModel) Complete(ctx context.Context, req *koine.LanguageRequest) (*koine.LanguageResponse, error) {
	s, err := m.Stream(ctx, req)
	if err != nil {
		return nil, err
	}
	return s.Collect()
}

// Stream performs one streaming Responses call.
func (m *LanguageModel) Stream(ctx context.Context, req *koine.LanguageRequest) (*koine.LanguageStream, error) {
	params, reqOpts, err := encodeResponsesRequest(m.model, req)
	if err != nil {
		return nil, wrapErr(err)
	}
	sse := m.client.Responses.NewStreaming(ctx, params, reqOpts...)

	var (
		pending []koine.Event
		items   = map[int64]responses.ResponseOutputItemUnion{}
		calls   = map[int64]*koine.ToolCallEvent{}
		parts   = map[int64]int64{}
		result  *responses.Response
	)

	next := func() (koine.Event, error) {
		for {
			if len(pending) > 0 {
				ev := pending[0]
				pending = pending[1:]
				return ev, nil
			}
			if result != nil {
				return koine.Event{}, io.EOF
			}
			if !sse.Next() {
				if err := sse.Err(); err != nil {
					return koine.Event{}, wrapErr(err)
				}
				return koine.Event{}, &koine.Error{Provider: Name, Message: "stream ended before the response completed"}
			}
			ev := sse.Current()
			index := int(ev.OutputIndex)
			switch ev.Type {
			case "response.output_item.added":
				if ev.Item.Type == "function_call" {
					tc := &koine.ToolCallEvent{ID: ev.Item.CallID, Name: ev.Item.Name}
					calls[ev.OutputIndex] = tc
					pending = append(pending, koine.Event{Type: koine.EventToolCallStart, Index: index, ToolCall: tc})
				}
			case "response.reasoning_summary_text.delta", "response.reasoning_text.delta":
				part := ev.SummaryIndex
				if ev.Type == "response.reasoning_text.delta" {
					part = ev.ContentIndex
				}
				if last, ok := parts[ev.OutputIndex]; ok && last != part {
					pending = append(pending, koine.Event{Type: koine.EventThinkingDelta, Index: index, Text: reasoningSeparator})
				}
				parts[ev.OutputIndex] = part
				pending = append(pending, koine.Event{Type: koine.EventThinkingDelta, Index: index, Text: ev.Delta})
			case "response.output_text.delta", "response.refusal.delta":
				pending = append(pending, koine.Event{Type: koine.EventTextDelta, Index: index, Text: ev.Delta})
			case "response.function_call_arguments.delta":
				if tc := calls[ev.OutputIndex]; tc != nil {
					pending = append(pending, koine.Event{
						Type:     koine.EventToolCallDelta,
						Index:    index,
						ToolCall: &koine.ToolCallEvent{ID: tc.ID, Name: tc.Name, InputDelta: ev.Delta},
					})
				}
			case "response.output_item.done":
				items[ev.OutputIndex] = ev.Item
				if ev.Item.Type == "function_call" {
					pending = append(pending, koine.Event{
						Type:     koine.EventToolCallEnd,
						Index:    index,
						ToolCall: &koine.ToolCallEvent{ID: ev.Item.CallID, Name: ev.Item.Name, Input: callInput(ev.Item)},
					})
				}
			case "response.completed", "response.incomplete":
				result = &ev.Response
				if u, ok := decodeResponsesUsage(result.Usage); ok {
					pending = append(pending, koine.Event{Type: koine.EventUsage, Usage: &u})
				}
				pending = append(pending, koine.Event{Type: koine.EventStop, StopReason: responsesStopReason(result, items)})
			case "response.failed":
				e := ev.Response.Error
				return koine.Event{}, &koine.Error{Provider: Name, Code: string(e.Code), Message: cmp.Or(e.Message, "response failed")}
			case "error":
				return koine.Event{}, &koine.Error{Provider: Name, Code: ev.Code, Message: cmp.Or(ev.Message, "response failed")}
			}
		}
	}
	final := func() (*koine.LanguageResponse, error) {
		return decodeResponsesResult(result, items), nil
	}
	return koine.NewLanguageStream(next, final, sse.Close), nil
}
