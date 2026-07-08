package openai

import (
	"context"
	"encoding/json"
	"io"
	"strings"

	"github.com/openai/openai-go/v3"

	"github.com/lf4096/koine"
)

// LanguageOptions carries OpenAI-compatibility request parameters. Pass it via
// LanguageRequest.ProviderOptions[openai.Name].
type LanguageOptions struct {
	// LegacyMaxTokens sends max_tokens instead of max_completion_tokens, for
	// compatible endpoints that predate the rename.
	LegacyMaxTokens bool
	// NoStreamUsage omits stream_options.include_usage, for compatible
	// endpoints that reject it.
	NoStreamUsage bool
	// ExtraBody merges additional top-level fields into the request body
	// (e.g. vendor thinking switches like enable_thinking).
	ExtraBody map[string]any
}

// LanguageModel speaks the OpenAI Chat Completions protocol. It is cheap to
// construct; build one per credential for multi-tenant use.
type LanguageModel struct {
	client openai.Client
}

// NewLanguageModel builds the chat model. Without options, credentials come
// from the environment (OPENAI_API_KEY), as the official SDK defines.
func NewLanguageModel(opts ...Option) *LanguageModel {
	return &LanguageModel{client: newClient(opts)}
}

func (m *LanguageModel) Name() string { return Name }

func (m *LanguageModel) Capabilities() koine.LanguageCapabilities {
	// Prompt caching is automatic on OpenAI; there is no cache control to
	// express, so CacheControl stays false.
	return koine.LanguageCapabilities{Thinking: true, ParallelToolCalls: true, Images: true, StructuredOutput: true}
}

// toolState assembles one streamed tool call; OpenAI deltas carry no block
// structure, so the provider synthesizes koine block indices itself.
type toolState struct {
	id    string
	name  string
	args  strings.Builder
	index int
	ended bool
}

// Complete performs one Chat Completions call and returns the final response.
func (m *LanguageModel) Complete(ctx context.Context, req *koine.LanguageRequest) (*koine.LanguageResponse, error) {
	s, err := m.Stream(ctx, req)
	if err != nil {
		return nil, err
	}
	return s.Collect()
}

// Stream performs one streaming Chat Completions call.
func (m *LanguageModel) Stream(ctx context.Context, req *koine.LanguageRequest) (*koine.LanguageStream, error) {
	params, reqOpts, err := encodeRequest(req)
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
			input := st.args.String()
			if input == "" {
				input = "{}"
			}
			pending = append(pending, koine.Event{
				Type:     koine.EventToolCallEnd,
				Index:    st.index,
				ToolCall: &koine.ToolCallEvent{ID: st.id, Name: st.name, Input: json.RawMessage(input)},
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
					pending = append(pending, koine.Event{Type: koine.EventStop, StopReason: mapFinishReason(finish)})
					continue
				}
				return koine.Event{}, io.EOF
			}
			chunk := sse.Current()
			acc.AddChunk(chunk)
			if u, ok := decodeUsage(chunk.Usage); ok {
				usage = u
				pending = append(pending, koine.Event{Type: koine.EventUsage, Usage: &u})
			}
			if len(chunk.Choices) == 0 {
				continue
			}
			choice := chunk.Choices[0]
			delta := choice.Delta
			if text, ok := reasoningDelta(delta); ok {
				reasoning.WriteString(text)
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
		return decodeResponse(&acc, reasoning.String(), tools, toolOrder, finish, usage), nil
	}
	return koine.NewLanguageStream(next, final, sse.Close), nil
}

// reasoningDelta reads the non-standard reasoning_content field that
// OpenAI-compatible reasoning models (DeepSeek, Qwen, GLM, Kimi) stream.
func reasoningDelta(delta openai.ChatCompletionChunkChoiceDelta) (string, bool) {
	// Extra fields carry status "invalid" in respjson (no schema to validate
	// against), so presence is tested via Raw, not Valid.
	field, ok := delta.JSON.ExtraFields["reasoning_content"]
	if !ok || field.Raw() == "" || field.Raw() == "null" {
		return "", false
	}
	var text string
	if err := json.Unmarshal([]byte(field.Raw()), &text); err != nil || text == "" {
		return "", false
	}
	return text, true
}
