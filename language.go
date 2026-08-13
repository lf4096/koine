package koine

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"iter"
	"strings"
)

// LanguageModel is one model spoken through one provider protocol, bound at
// construction. Implementations only translate formats; transport stays in
// the official provider SDK underneath.
type LanguageModel interface {
	Model() string
	Provider() string
	Capabilities() LanguageCapabilities
	// Complete performs one call and returns the final response.
	Complete(ctx context.Context, req *LanguageRequest) (*LanguageResponse, error)
	// Stream performs one call and returns the events as they arrive.
	Stream(ctx context.Context, req *LanguageRequest) (*LanguageStream, error)
}

// LanguageCapabilities reports what a provider can express, so callers can
// degrade before sending a request the provider would reject or silently
// ignore.
type LanguageCapabilities struct {
	Thinking          bool
	CacheControl      bool
	ParallelToolCalls bool
	Images            bool
	// StructuredOutput reports native schema-constrained output support at
	// the protocol level; individual models may still reject it.
	StructuredOutput bool
}

// LanguageRequest is one model call. It carries only the surface an agent loop
// consumes; provider-specific parameters travel in ProviderOptions.
type LanguageRequest struct {
	System      string
	Messages    []Message
	Tools       []Tool
	ToolChoice  *ToolChoice
	MaxTokens   int
	Temperature *float64
	TopP        *float64
	// StopSequences end generation as soon as the model emits any of them.
	// The matched sequence is excluded from the output, and the response
	// reports StopStopSequence.
	StopSequences []string
	// Thinking enables reasoning. nil leaves the provider default (usually off).
	Thinking *Thinking
	// ResponseFormat constrains the output to JSON where the provider supports
	// it natively. Providers without support return an error rather than
	// silently dropping the constraint.
	ResponseFormat *ResponseFormat
	// CacheRetention controls prompt caching where the provider exposes it.
	CacheRetention CacheRetention
	// ProviderOptions passes provider-specific parameters, keyed by provider
	// name. Values are the LanguageOptions struct exported by that provider
	// package.
	ProviderOptions map[string]any
}

// Message is one conversation turn in the canonical model.
type Message struct {
	Role   Role   `json:"role"`
	Blocks Blocks `json:"blocks"`
}

// UserText builds a user message with a single text block.
func UserText(text string) Message {
	return Message{Role: RoleUser, Blocks: Blocks{&TextBlock{Text: text}}}
}

// ToolResultText builds a tool message answering toolUseID with text.
func ToolResultText(toolUseID, text string, isError bool) Message {
	return Message{Role: RoleTool, Blocks: Blocks{&ToolResultBlock{
		ToolUseID: toolUseID,
		Content:   Blocks{&TextBlock{Text: text}},
		IsError:   isError,
	}}}
}

// ToolResultJSON builds a tool message answering toolUseID with a structured
// JSON result.
func ToolResultJSON(toolUseID string, result json.RawMessage, isError bool) Message {
	return Message{Role: RoleTool, Blocks: Blocks{&ToolResultBlock{
		ToolUseID: toolUseID,
		Result:    result,
		IsError:   isError,
	}}}
}

// Text concatenates the message's text blocks.
func (m Message) Text() string {
	var sb strings.Builder
	for _, b := range m.Blocks {
		if t, ok := b.(*TextBlock); ok {
			sb.WriteString(t.Text)
		}
	}
	return sb.String()
}

// ToolUses returns the message's tool_use blocks in order.
func (m Message) ToolUses() []*ToolUseBlock {
	var out []*ToolUseBlock
	for _, b := range m.Blocks {
		if t, ok := b.(*ToolUseBlock); ok {
			out = append(out, t)
		}
	}
	return out
}

// Role identifies who produced a message.
type Role string

const (
	RoleUser      Role = "user"
	RoleAssistant Role = "assistant"
	// RoleTool carries tool results back to the model. Providers reshape it
	// to their protocol's native form.
	RoleTool Role = "tool"
)

// Blocks is a Block list that marshals each element with a "type"
// discriminator, making messages directly usable as a persistence format.
type Blocks []Block

func (bs Blocks) MarshalJSON() ([]byte, error) {
	out := make([]json.RawMessage, len(bs))
	for i, b := range bs {
		raw, err := marshalBlock(b)
		if err != nil {
			return nil, err
		}
		out[i] = raw
	}
	return json.Marshal(out)
}

func (bs *Blocks) UnmarshalJSON(data []byte) error {
	var raws []json.RawMessage
	if err := json.Unmarshal(data, &raws); err != nil {
		return err
	}
	blocks := make(Blocks, len(raws))
	for i, raw := range raws {
		b, err := unmarshalBlock(raw)
		if err != nil {
			return err
		}
		blocks[i] = b
	}
	*bs = blocks
	return nil
}

func marshalBlock(b Block) (json.RawMessage, error) {
	body, err := json.Marshal(b)
	if err != nil {
		return nil, err
	}
	head := `{"type":"` + string(b.BlockType()) + `"`
	if len(body) <= 2 {
		return json.RawMessage(head + "}"), nil
	}
	return json.RawMessage(head + "," + string(body[1:])), nil
}

func unmarshalBlock(data json.RawMessage) (Block, error) {
	var head struct {
		Type BlockType `json:"type"`
	}
	if err := json.Unmarshal(data, &head); err != nil {
		return nil, err
	}
	var b Block
	switch head.Type {
	case BlockText:
		b = &TextBlock{}
	case BlockThinking:
		b = &ThinkingBlock{}
	case BlockToolUse:
		b = &ToolUseBlock{}
	case BlockToolResult:
		b = &ToolResultBlock{}
	case BlockImage:
		b = &ImageBlock{}
	default:
		return nil, fmt.Errorf("koine: unknown block type %q", head.Type)
	}
	if err := json.Unmarshal(data, b); err != nil {
		return nil, err
	}
	return b, nil
}

// Block is one unit of message content. Concrete types: TextBlock,
// ThinkingBlock, ToolUseBlock, ToolResultBlock, ImageBlock.
type Block interface {
	BlockType() BlockType
}

// BlockType discriminates the concrete Block implementations.
type BlockType string

const (
	BlockText       BlockType = "text"
	BlockThinking   BlockType = "thinking"
	BlockToolUse    BlockType = "tool_use"
	BlockToolResult BlockType = "tool_result"
	BlockImage      BlockType = "image"
)

// TextBlock is plain text content. Raw is set when the provider attaches
// opaque metadata to text (Gemini thought signatures ride on text parts).
type TextBlock struct {
	Text string       `json:"text"`
	Raw  *ProviderRaw `json:"raw,omitempty"`
}

func (*TextBlock) BlockType() BlockType { return BlockText }

// ThinkingBlock is model reasoning output. Redacted thinking has empty Text;
// its opaque payload lives in Raw and is only replayable via Raw.
type ThinkingBlock struct {
	Text      string       `json:"text,omitempty"`
	Signature string       `json:"signature,omitempty"`
	Redacted  bool         `json:"redacted,omitempty"`
	Raw       *ProviderRaw `json:"raw,omitempty"`
}

func (*ThinkingBlock) BlockType() BlockType { return BlockThinking }

// ToolUseBlock is a model-issued tool call. Input is the raw JSON arguments
// as generated by the model (possibly malformed; validate before use).
type ToolUseBlock struct {
	ID    string          `json:"id"`
	Name  string          `json:"name"`
	Input json.RawMessage `json:"input"`
	Raw   *ProviderRaw    `json:"raw,omitempty"`
}

func (*ToolUseBlock) BlockType() BlockType { return BlockToolUse }

// ToolResultBlock answers a prior ToolUseBlock. It appears only in RoleTool
// messages. Exactly one of Content and Result carries the payload: Content
// for text blocks, plus image blocks on providers whose protocol takes them
// (others return an error rather than dropping result content); Result for a
// structured JSON value (delivered natively where the provider accepts
// objects, serialized as text elsewhere).
type ToolResultBlock struct {
	ToolUseID string          `json:"tool_use_id"`
	Content   Blocks          `json:"content,omitempty"`
	Result    json.RawMessage `json:"result,omitempty"`
	IsError   bool            `json:"is_error,omitempty"`
}

func (*ToolResultBlock) BlockType() BlockType { return BlockToolResult }

// Tool declares a function the model may call.
type Tool struct {
	Name        string
	Description string
	// InputSchema is the tool's parameter JSON Schema. Schemas are static
	// configuration, typed as a map so each request encodes without reparsing.
	InputSchema map[string]any
}

// ToolChoice constrains tool use for one request.
type ToolChoice struct {
	Mode ToolChoiceMode
	// Name is the forced tool when Mode is ToolChoiceTool.
	Name string
}

// ToolChoiceMode selects how the model may use tools.
type ToolChoiceMode string

const (
	ToolChoiceAuto     ToolChoiceMode = "auto"
	ToolChoiceNone     ToolChoiceMode = "none"
	ToolChoiceRequired ToolChoiceMode = "required"
	// ToolChoiceTool forces the named tool.
	ToolChoiceTool ToolChoiceMode = "tool"
)

// Thinking configures reasoning for one request. A zero-value Thinking (no
// Effort, no BudgetTokens) leaves the provider default, same as nil.
// Providers send whichever field matches their native dial and convert the
// other; where both dials are native (Gemini), BudgetTokens wins.
type Thinking struct {
	Effort       ThinkingEffort
	BudgetTokens int
}

// ThinkingEffort is a normalized reasoning-effort level. Providers map it to
// their native knob: an effort level or a token budget. On Gemini the knob is
// thinkingLevel, which Gemini 2.x models reject; use BudgetTokens for those.
type ThinkingEffort string

const (
	// ThinkingNone disables reasoning. Models that cannot switch reasoning
	// off reject the request; combining it with BudgetTokens is an error.
	ThinkingNone    ThinkingEffort = "none"
	ThinkingMinimal ThinkingEffort = "minimal"
	ThinkingLow     ThinkingEffort = "low"
	ThinkingMedium  ThinkingEffort = "medium"
	ThinkingHigh    ThinkingEffort = "high"
	ThinkingXHigh   ThinkingEffort = "xhigh"
	ThinkingMax     ThinkingEffort = "max"
)

// ResponseFormat requests schema-constrained JSON output.
type ResponseFormat struct {
	// Schema is a JSON Schema for the output. Empty requests a schemaless
	// JSON mode where the provider has one, and errors where it does not.
	Schema map[string]any
	// Name labels the format for providers that require one; empty gets a
	// default. Description optionally explains the format to the model.
	Name        string
	Description string
	// Strict enforces exact schema adherence on providers where enforcement
	// is opt-in; providers that always enforce ignore it.
	Strict bool
}

// CacheRetention is a normalized prompt-cache preference. Providers with
// explicit cache controls place breakpoints automatically; providers with
// implicit caching ignore it.
type CacheRetention string

const (
	// CacheDefault leaves the provider's default behavior.
	CacheDefault CacheRetention = ""
	// CacheShort requests short-lived caching (typically minutes).
	CacheShort CacheRetention = "short"
	// CacheLong requests long-lived caching (typically an hour).
	CacheLong CacheRetention = "long"
)

// LanguageResponse is the final result of one call. Message blocks carry
// ProviderRaw, so appending it to LanguageRequest.Messages for the next turn
// round-trips provider fidelity.
type LanguageResponse struct {
	Message    Message    `json:"message"`
	StopReason StopReason `json:"stop_reason"`
	Usage      Usage      `json:"usage"`
	// Model is the provider-reported model that served the request.
	Model    string `json:"model,omitempty"`
	Provider string `json:"provider,omitempty"`
}

// StopReason is the normalized reason a response ended.
type StopReason string

const (
	StopEndTurn       StopReason = "end_turn"
	StopToolUse       StopReason = "tool_use"
	StopMaxTokens     StopReason = "max_tokens"
	StopStopSequence  StopReason = "stop_sequence"
	StopContentFilter StopReason = "content_filter"
	StopOther         StopReason = "other"
)

// LanguageStream delivers unified events for one call. Consume with a range
// loop over Events (or step with Next/Event); after iteration ends, check
// Err, then read the accumulated final response.
type LanguageStream struct {
	next    func() (Event, error)
	final   func() (*LanguageResponse, error)
	closeFn func() error

	event  Event
	resp   *LanguageResponse
	err    error
	done   bool
	closed bool
}

// NewLanguageStream assembles a LanguageStream from provider callbacks: next
// returns events until io.EOF; final produces the accumulated response after
// EOF; closeFn releases the underlying connection.
func NewLanguageStream(next func() (Event, error), final func() (*LanguageResponse, error), closeFn func() error) *LanguageStream {
	return &LanguageStream{next: next, final: final, closeFn: closeFn}
}

// Collect drains the stream, closes it, and returns the final response.
func (s *LanguageStream) Collect() (*LanguageResponse, error) {
	defer s.Close()
	for s.Next() {
	}
	if err := s.Err(); err != nil {
		return nil, err
	}
	return s.Response(), nil
}

// Events returns an iterator over the stream's events, for use with a range
// loop. Iteration ends at end of stream, on error, or on break; check Err
// after the loop, as with Next.
func (s *LanguageStream) Events() iter.Seq[Event] {
	return func(yield func(Event) bool) {
		for s.Next() {
			if !yield(s.event) {
				return
			}
		}
	}
}

// Next advances to the next event, returning false at end of stream or error.
func (s *LanguageStream) Next() bool {
	if s.done {
		return false
	}
	ev, err := s.next()
	if err != nil {
		s.done = true
		if errors.Is(err, io.EOF) {
			s.resp, s.err = s.final()
		} else {
			s.err = err
		}
		return false
	}
	s.event = ev
	return true
}

// Event returns the current event. Valid after Next returns true.
func (s *LanguageStream) Event() Event { return s.event }

// Err returns the terminal error, if any. Valid after Next returns false.
func (s *LanguageStream) Err() error { return s.err }

// Response returns the final accumulated response. Valid after Next returns
// false with nil Err.
func (s *LanguageStream) Response() *LanguageResponse { return s.resp }

// Close releases the stream. Safe to call multiple times.
func (s *LanguageStream) Close() error {
	if s.closed {
		return nil
	}
	s.closed = true
	s.done = true
	if s.closeFn != nil {
		return s.closeFn()
	}
	return nil
}

// Event is one unified stream increment. Index is the content-block ordinal
// within the response; a change of Index marks a block boundary.
type Event struct {
	Type  EventType
	Index int
	// Text carries the delta for EventTextDelta and EventThinkingDelta.
	Text string
	// ToolCall is set on the tool_call_* events.
	ToolCall *ToolCallEvent
	// Usage is set on EventUsage.
	Usage *Usage
	// StopReason is set on EventStop.
	StopReason StopReason
}

// EventType discriminates stream events.
type EventType string

const (
	EventTextDelta     EventType = "text_delta"
	EventThinkingDelta EventType = "thinking_delta"
	EventToolCallStart EventType = "tool_call_start"
	EventToolCallDelta EventType = "tool_call_delta"
	EventToolCallEnd   EventType = "tool_call_end"
	EventUsage         EventType = "usage"
	EventStop          EventType = "stop"
)

// ToolCallEvent carries tool-call streaming state. ID and Name are set from
// tool_call_start on; InputDelta carries argument JSON fragments on
// tool_call_delta; Input is the complete argument JSON on tool_call_end.
type ToolCallEvent struct {
	ID         string
	Name       string
	InputDelta string
	Input      json.RawMessage
}
