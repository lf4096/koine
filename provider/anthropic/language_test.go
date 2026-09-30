package anthropic

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"testing"

	"github.com/anthropics/anthropic-sdk-go"

	"github.com/lf4096/koine"
)

func sseServer(t *testing.T, capture *json.RawMessage, events ...string) *httptest.Server {
	t.Helper()
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body json.RawMessage
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Errorf("decode request body: %v", err)
		}
		*capture = body
		w.Header().Set("Content-Type", "text/event-stream")
		for _, e := range events {
			fmt.Fprint(w, e)
		}
	}))
	t.Cleanup(server.Close)
	return server
}

func event(name, data string) string {
	return "event: " + name + "\ndata: " + data + "\n\n"
}

var streamFixture = []string{
	event("message_start", `{"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant","model":"claude-test","content":[],"stop_reason":null,"usage":{"input_tokens":25,"output_tokens":1,"cache_creation_input_tokens":2,"cache_read_input_tokens":3}}}`),
	event("content_block_start", `{"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":"","signature":""}}`),
	event("content_block_delta", `{"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"Let me think"}}`),
	event("content_block_delta", `{"type":"content_block_delta","index":0,"delta":{"type":"signature_delta","signature":"SIG123"}}`),
	event("content_block_stop", `{"type":"content_block_stop","index":0}`),
	event("content_block_start", `{"type":"content_block_start","index":1,"content_block":{"type":"text","text":""}}`),
	event("content_block_delta", `{"type":"content_block_delta","index":1,"delta":{"type":"text_delta","text":"Hello"}}`),
	event("content_block_delta", `{"type":"content_block_delta","index":1,"delta":{"type":"text_delta","text":" world"}}`),
	event("content_block_stop", `{"type":"content_block_stop","index":1}`),
	event("content_block_start", `{"type":"content_block_start","index":2,"content_block":{"type":"tool_use","id":"toolu_1","name":"get_weather","input":{}}}`),
	event("content_block_delta", `{"type":"content_block_delta","index":2,"delta":{"type":"input_json_delta","partial_json":"{\"city\":"}}`),
	event("content_block_delta", `{"type":"content_block_delta","index":2,"delta":{"type":"input_json_delta","partial_json":"\"SF\"}"}}`),
	event("content_block_stop", `{"type":"content_block_stop","index":2}`),
	event("message_delta", `{"type":"message_delta","delta":{"stop_reason":"tool_use","stop_sequence":null},"usage":{"input_tokens":25,"output_tokens":42,"cache_creation_input_tokens":2,"cache_read_input_tokens":3,"output_tokens_details":{"thinking_tokens":7}}}`),
	event("message_stop", `{"type":"message_stop"}`),
}

func TestStreamEventsAndResponse(t *testing.T) {
	var body json.RawMessage
	server := sseServer(t, &body, streamFixture...)
	d := New(WithAPIKey("test"), WithBaseURL(server.URL)).LanguageModel("claude-test")

	stream, err := d.Stream(context.Background(), &koine.LanguageRequest{
		Messages: []koine.Message{koine.UserText("hi")},
	})
	if err != nil {
		t.Fatal(err)
	}
	defer stream.Close()

	var events []koine.Event
	for stream.Next() {
		events = append(events, stream.Event())
	}
	if err := stream.Err(); err != nil {
		t.Fatal(err)
	}

	wantTypes := []koine.EventType{
		koine.EventThinkingDelta,
		koine.EventTextDelta, koine.EventTextDelta,
		koine.EventToolCallStart, koine.EventToolCallDelta, koine.EventToolCallDelta, koine.EventToolCallEnd,
		koine.EventUsage, koine.EventStop,
	}
	var gotTypes []koine.EventType
	for _, e := range events {
		gotTypes = append(gotTypes, e.Type)
	}
	if !reflect.DeepEqual(gotTypes, wantTypes) {
		t.Fatalf("event types = %v, want %v", gotTypes, wantTypes)
	}
	end := events[6]
	if end.ToolCall.ID != "toolu_1" || end.ToolCall.Name != "get_weather" || string(end.ToolCall.Input) != `{"city":"SF"}` {
		t.Errorf("tool_call_end = %#v", end.ToolCall)
	}
	if stop := events[8]; stop.StopReason != koine.StopToolUse {
		t.Errorf("stop reason = %q", stop.StopReason)
	}

	resp := stream.Response()
	if resp == nil {
		t.Fatal("nil response")
	}
	if resp.Model != "claude-test" || resp.Provider != Name || resp.StopReason != koine.StopToolUse {
		t.Errorf("response meta = %#v", resp)
	}
	wantUsage := koine.Usage{InputTokens: 25, OutputTokens: 42, CacheReadTokens: 3, CacheWriteTokens: 2, ReasoningTokens: 7}
	if resp.Usage != wantUsage {
		t.Errorf("usage = %#v, want %#v", resp.Usage, wantUsage)
	}
	blocks := resp.Message.Blocks
	if len(blocks) != 3 {
		t.Fatalf("blocks = %#v", blocks)
	}
	thinking := blocks[0].(*koine.ThinkingBlock)
	if thinking.Text != "Let me think" || thinking.Signature != "SIG123" {
		t.Errorf("thinking = %#v", thinking)
	}
	if thinking.Raw == nil || thinking.Raw.Provider != Name {
		t.Errorf("thinking raw = %#v", thinking.Raw)
	}
	if text := blocks[1].(*koine.TextBlock); text.Text != "Hello world" {
		t.Errorf("text = %#v", text)
	}
	tool := blocks[2].(*koine.ToolUseBlock)
	if tool.ID != "toolu_1" || string(tool.Input) != `{"city":"SF"}` || tool.Raw == nil {
		t.Errorf("tool = %#v", tool)
	}
}

func TestEncodeRequestWire(t *testing.T) {
	var body json.RawMessage
	server := sseServer(t, &body, streamFixture...)
	d := New(WithAPIKey("test"), WithBaseURL(server.URL)).LanguageModel("claude-test")

	topK := 5
	temp := 0.5
	req := &koine.LanguageRequest{
		System:         "be brief",
		MaxTokens:      1000,
		Temperature:    &temp,
		Thinking:       &koine.Thinking{Effort: koine.ThinkingHigh},
		CacheRetention: koine.CacheLong,
		Tools: []koine.Tool{{
			Name:        "get_weather",
			Description: "weather lookup",
			InputSchema: map[string]any{
				"type":                 "object",
				"properties":           map[string]any{"city": map[string]any{"type": "string"}},
				"required":             []string{"city"},
				"additionalProperties": false,
			},
		}},
		ToolChoice: &koine.ToolChoice{Mode: koine.ToolChoiceRequired},
		Messages: []koine.Message{
			koine.UserText("hi"),
			{Role: koine.RoleAssistant, Blocks: koine.Blocks{
				&koine.ThinkingBlock{Text: "T", Signature: "S", Raw: &koine.ProviderRaw{Provider: Name, JSON: json.RawMessage(`{"type":"thinking","thinking":"T","signature":"S"}`)}},
				&koine.ToolUseBlock{ID: "toolu_1", Name: "get_weather", Input: json.RawMessage(`{"city":"SF"}`)},
			}},
			koine.ToolResultText("toolu_1", "sunny", false),
		},
		ProviderOptions: map[string]any{Name: LanguageOptions{TopK: &topK}},
	}
	stream, err := d.Stream(context.Background(), req)
	if err != nil {
		t.Fatal(err)
	}
	for stream.Next() {
	}
	stream.Close()

	var wire struct {
		Model     string  `json:"model"`
		MaxTokens int     `json:"max_tokens"`
		TopK      int     `json:"top_k"`
		Temp      float64 `json:"temperature"`
		System    []struct {
			Text string `json:"text"`
		} `json:"system"`
		Thinking struct {
			Type         string `json:"type"`
			BudgetTokens int    `json:"budget_tokens"`
		} `json:"thinking"`
		CacheControl struct {
			Type string `json:"type"`
			TTL  string `json:"ttl"`
		} `json:"cache_control"`
		ToolChoice struct {
			Type string `json:"type"`
		} `json:"tool_choice"`
		Tools []struct {
			Name        string         `json:"name"`
			Description string         `json:"description"`
			InputSchema map[string]any `json:"input_schema"`
		} `json:"tools"`
		Messages []struct {
			Role    string            `json:"role"`
			Content []json.RawMessage `json:"content"`
		} `json:"messages"`
	}
	if err := json.Unmarshal(body, &wire); err != nil {
		t.Fatalf("decode wire: %v\n%s", err, body)
	}
	if wire.Model != "claude-test" || wire.TopK != 5 || wire.Temp != 0.5 || wire.System[0].Text != "be brief" {
		t.Errorf("basics: %+v", wire)
	}
	// MaxTokens must exceed the thinking budget.
	if wire.Thinking.Type != "enabled" || wire.Thinking.BudgetTokens != 16384 || wire.MaxTokens != 1000+16384 {
		t.Errorf("thinking: %+v max_tokens=%d", wire.Thinking, wire.MaxTokens)
	}
	if wire.CacheControl.Type != "ephemeral" || wire.CacheControl.TTL != "1h" {
		t.Errorf("cache_control: %+v", wire.CacheControl)
	}
	if wire.ToolChoice.Type != "any" {
		t.Errorf("tool_choice: %+v", wire.ToolChoice)
	}
	tool := wire.Tools[0]
	if tool.Name != "get_weather" || tool.Description != "weather lookup" {
		t.Errorf("tool: %+v", tool)
	}
	if tool.InputSchema["type"] != "object" || tool.InputSchema["additionalProperties"] != false {
		t.Errorf("tool schema: %+v", tool.InputSchema)
	}
	if len(wire.Messages) != 3 {
		t.Fatalf("messages: %d\n%s", len(wire.Messages), body)
	}
	// Same-provider thinking blocks must round-trip verbatim.
	var thinkingWire map[string]any
	if err := json.Unmarshal(wire.Messages[1].Content[0], &thinkingWire); err != nil {
		t.Fatal(err)
	}
	want := map[string]any{"type": "thinking", "thinking": "T", "signature": "S"}
	if !reflect.DeepEqual(thinkingWire, want) {
		t.Errorf("thinking resend = %v, want %v", thinkingWire, want)
	}
	// The tool role becomes a user message carrying tool_result blocks.
	if wire.Messages[2].Role != "user" {
		t.Errorf("tool message role = %q", wire.Messages[2].Role)
	}
	var toolResult struct {
		Type      string `json:"type"`
		ToolUseID string `json:"tool_use_id"`
	}
	if err := json.Unmarshal(wire.Messages[2].Content[0], &toolResult); err != nil {
		t.Fatal(err)
	}
	if toolResult.Type != "tool_result" || toolResult.ToolUseID != "toolu_1" {
		t.Errorf("tool_result = %+v", toolResult)
	}
}

func TestForeignThinkingDropped(t *testing.T) {
	var body json.RawMessage
	server := sseServer(t, &body, streamFixture...)
	d := New(WithAPIKey("test"), WithBaseURL(server.URL)).LanguageModel("claude-test")

	req := &koine.LanguageRequest{
		Messages: []koine.Message{
			koine.UserText("hi"),
			{Role: koine.RoleAssistant, Blocks: koine.Blocks{
				&koine.ThinkingBlock{Text: "foreign reasoning"},
				&koine.TextBlock{Text: "answer"},
			}},
			koine.UserText("next"),
		},
	}
	stream, err := d.Stream(context.Background(), req)
	if err != nil {
		t.Fatal(err)
	}
	for stream.Next() {
	}
	stream.Close()

	var wire struct {
		Messages []struct {
			Content []struct {
				Type string `json:"type"`
			} `json:"content"`
		} `json:"messages"`
	}
	if err := json.Unmarshal(body, &wire); err != nil {
		t.Fatal(err)
	}
	assistant := wire.Messages[1]
	if len(assistant.Content) != 1 || assistant.Content[0].Type != "text" {
		t.Errorf("assistant content = %+v, want text only", assistant.Content)
	}
}

func TestStopReasons(t *testing.T) {
	cases := map[string]koine.StopReason{
		"end_turn":                      koine.StopEndTurn,
		"max_tokens":                    koine.StopMaxTokens,
		"model_context_window_exceeded": koine.StopMaxTokens,
		"refusal":                       koine.StopContentFilter,
		"something_new":                 koine.StopOther,
	}
	for in, want := range cases {
		if got := mapStopReason(anthropic.StopReason(in)); got != want {
			t.Errorf("mapStopReason(%q) = %q, want %q", in, got, want)
		}
	}
}

func TestErrorMapping(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusTooManyRequests)
		fmt.Fprint(w, `{"type":"error","error":{"type":"rate_limit_error","message":"slow down"}}`)
	}))
	defer server.Close()
	d := New(WithAPIKey("test"), WithBaseURL(server.URL)).LanguageModel("claude-test")

	stream, err := d.Stream(context.Background(), &koine.LanguageRequest{
		Messages: []koine.Message{koine.UserText("hi")},
	})
	if err != nil {
		t.Fatal(err)
	}
	for stream.Next() {
	}
	kerr, ok := errors.AsType[*koine.Error](stream.Err())
	if !ok {
		t.Fatalf("Err() = %v, want *koine.Error", stream.Err())
	}
	if kerr.Status != http.StatusTooManyRequests || kerr.Provider != Name || !kerr.Retryable() {
		t.Errorf("error = %#v", kerr)
	}
	if kerr.Message != "slow down" {
		t.Errorf("message = %q, want clean provider message", kerr.Message)
	}
}

func TestStructuredOutputWire(t *testing.T) {
	var body json.RawMessage
	server := sseServer(t, &body, streamFixture...)
	d := New(WithAPIKey("test"), WithBaseURL(server.URL)).LanguageModel("claude-test")

	stream, err := d.Stream(context.Background(), &koine.LanguageRequest{
		Messages: []koine.Message{koine.UserText("hi")},
		ResponseFormat: &koine.ResponseFormat{
			Schema: map[string]any{"type": "object", "properties": map[string]any{"answer": map[string]any{"type": "string"}}},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	for stream.Next() {
	}
	stream.Close()

	var wire struct {
		OutputConfig struct {
			Format struct {
				Type   string         `json:"type"`
				Schema map[string]any `json:"schema"`
			} `json:"format"`
		} `json:"output_config"`
	}
	if err := json.Unmarshal(body, &wire); err != nil {
		t.Fatal(err)
	}
	if wire.OutputConfig.Format.Type != "json_schema" || wire.OutputConfig.Format.Schema["type"] != "object" {
		t.Errorf("output_config = %+v", wire.OutputConfig)
	}
}

func TestStructuredOutputRequiresSchema(t *testing.T) {
	d := New(WithAPIKey("test")).LanguageModel("claude-test")
	_, err := d.Stream(context.Background(), &koine.LanguageRequest{
		Messages:       []koine.Message{koine.UserText("hi")},
		ResponseFormat: &koine.ResponseFormat{},
	})
	if err == nil || !strings.Contains(err.Error(), "requires a schema") {
		t.Errorf("err = %v", err)
	}
}

func TestToolResultJSONWire(t *testing.T) {
	var body json.RawMessage
	server := sseServer(t, &body, streamFixture...)
	d := New(WithAPIKey("test"), WithBaseURL(server.URL)).LanguageModel("claude-test")

	stream, err := d.Stream(context.Background(), &koine.LanguageRequest{
		Messages: []koine.Message{
			koine.UserText("hi"),
			{Role: koine.RoleAssistant, Blocks: koine.Blocks{
				&koine.ToolUseBlock{ID: "toolu_1", Name: "get_weather", Input: json.RawMessage(`{}`)},
			}},
			koine.ToolResultJSON("toolu_1", json.RawMessage(`{"temp":22}`), false),
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	for stream.Next() {
	}
	stream.Close()

	var wire struct {
		Messages []struct {
			Content []struct {
				Type    string `json:"type"`
				Content []struct {
					Text string `json:"text"`
				} `json:"content"`
			} `json:"content"`
		} `json:"messages"`
	}
	if err := json.Unmarshal(body, &wire); err != nil {
		t.Fatal(err)
	}
	tr := wire.Messages[2].Content[0]
	if tr.Type != "tool_result" || len(tr.Content) != 1 || tr.Content[0].Text != `{"temp":22}` {
		t.Errorf("tool_result = %+v", tr)
	}
}

func TestThinkingWire(t *testing.T) {
	var body json.RawMessage
	server := sseServer(t, &body, streamFixture...)
	d := New(WithAPIKey("test"), WithBaseURL(server.URL)).LanguageModel("claude-test")

	if _, err := d.Stream(context.Background(), &koine.LanguageRequest{
		Thinking: &koine.Thinking{Effort: koine.ThinkingNone, BudgetTokens: 100},
		Messages: []koine.Message{koine.UserText("hi")},
	}); err == nil {
		t.Fatal("ThinkingNone with BudgetTokens: want error")
	}

	stream, err := d.Stream(context.Background(), &koine.LanguageRequest{
		MaxTokens: 100,
		Thinking:  &koine.Thinking{Effort: koine.ThinkingNone},
		Messages:  []koine.Message{koine.UserText("hi")},
	})
	if err != nil {
		t.Fatal(err)
	}
	for stream.Next() {
	}
	stream.Close()

	var wire struct {
		Thinking struct {
			Type string `json:"type"`
		} `json:"thinking"`
		MaxTokens int `json:"max_tokens"`
	}
	if err := json.Unmarshal(body, &wire); err != nil {
		t.Fatalf("decode wire: %v\n%s", err, body)
	}
	if wire.Thinking.Type != "disabled" || wire.MaxTokens != 100 {
		t.Errorf("wire: %+v", wire)
	}
}

func TestThinkingOverrideWire(t *testing.T) {
	adaptive := anthropic.ThinkingConfigParamUnion{OfAdaptive: &anthropic.ThinkingConfigAdaptiveParam{Display: anthropic.ThinkingConfigAdaptiveDisplayOmitted}}
	betweenTools := anthropic.ThinkingConfigParamUnion{OfBetweenTools: &anthropic.ThinkingConfigBetweenToolsParam{}}
	enabled := anthropic.ThinkingConfigParamOfEnabled(8000)
	cases := []struct {
		name          string
		maxTokens     int
		opts          LanguageOptions
		format        *koine.ResponseFormat
		wantThinking  string
		wantOutput    string
		wantMaxTokens int
	}{
		{"adaptive replaces normalized thinking", 100, LanguageOptions{Thinking: &adaptive, Effort: anthropic.OutputConfigEffortXhigh}, nil, `{"type":"adaptive","display":"omitted"}`, `{"effort":"xhigh"}`, 100},
		{"effort merges with format", 100, LanguageOptions{Thinking: &adaptive, Effort: anthropic.OutputConfigEffortLow}, &koine.ResponseFormat{Schema: map[string]any{"type": "object"}}, `{"type":"adaptive","display":"omitted"}`, `{"effort":"low","format":{"schema":{"type":"object"},"type":"json_schema"}}`, 100},
		{"between tools", 100, LanguageOptions{Thinking: &betweenTools}, nil, `{"type":"between_tools"}`, ``, 100},
		{"enabled budget still raises max tokens", 0, LanguageOptions{Thinking: &enabled}, nil, `{"budget_tokens":8000,"type":"enabled"}`, ``, defaultMaxTokens + 8000},
	}
	for _, c := range cases {
		var body json.RawMessage
		server := sseServer(t, &body, streamFixture...)
		d := New(WithAPIKey("test"), WithBaseURL(server.URL)).LanguageModel("claude-test")
		stream, err := d.Stream(context.Background(), &koine.LanguageRequest{
			MaxTokens:       c.maxTokens,
			Thinking:        &koine.Thinking{Effort: koine.ThinkingHigh},
			ResponseFormat:  c.format,
			Messages:        []koine.Message{koine.UserText("hi")},
			ProviderOptions: map[string]any{Name: c.opts},
		})
		if err != nil {
			t.Fatalf("%s: %v", c.name, err)
		}
		for stream.Next() {
		}
		stream.Close()

		var wire struct {
			Thinking     json.RawMessage `json:"thinking"`
			OutputConfig json.RawMessage `json:"output_config"`
			MaxTokens    int             `json:"max_tokens"`
		}
		if err := json.Unmarshal(body, &wire); err != nil {
			t.Fatalf("%s: decode wire: %v", c.name, err)
		}
		if !jsonEqual(wire.Thinking, c.wantThinking) || !jsonEqual(wire.OutputConfig, c.wantOutput) || wire.MaxTokens != c.wantMaxTokens {
			t.Errorf("%s: thinking = %s, output_config = %s, max_tokens = %d", c.name, wire.Thinking, wire.OutputConfig, wire.MaxTokens)
		}
	}
}

func jsonEqual(got json.RawMessage, want string) bool {
	if len(got) == 0 || want == "" {
		return len(got) == 0 && want == ""
	}
	var g, w any
	return json.Unmarshal(got, &g) == nil && json.Unmarshal([]byte(want), &w) == nil && reflect.DeepEqual(g, w)
}
