package openai

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

	"github.com/openai/openai-go/v3"

	"github.com/lf4096/koine"
)

func chatServer(t *testing.T, capture *json.RawMessage, chunks ...string) *httptest.Server {
	t.Helper()
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body json.RawMessage
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Errorf("decode request body: %v", err)
		}
		*capture = body
		w.Header().Set("Content-Type", "text/event-stream")
		// Endpoints interleave comment-only keepalives, which carry no event.
		fmt.Fprint(w, ": keepalive\n\n")
		for _, c := range chunks {
			fmt.Fprintf(w, "data: %s\n\n", c)
		}
		fmt.Fprint(w, "data: [DONE]\n\n")
	}))
	t.Cleanup(server.Close)
	return server
}

func chatChunk(payload string) string {
	return `{"id":"c1","object":"chat.completion.chunk","created":1,"model":"deepseek-test",` + payload + `}`
}

var chatFixture = []string{
	chatChunk(`"choices":[{"index":0,"delta":{"role":"assistant","reasoning_content":"pondering"},"finish_reason":null}]`),
	chatChunk(`"choices":[{"index":0,"delta":{"content":"Hi"},"finish_reason":null}]`),
	chatChunk(`"choices":[{"index":0,"delta":{"content":" there"},"finish_reason":null}]`),
	chatChunk(`"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"get_weather","arguments":""}}]},"finish_reason":null}]`),
	chatChunk(`"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"city\":\"SF\"}"}}]},"finish_reason":null}]`),
	chatChunk(`"choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]`),
	chatChunk(`"choices":[],"usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15,"prompt_tokens_details":{"cached_tokens":2},"completion_tokens_details":{"reasoning_tokens":3}}`),
}

func TestChatAssistantImageDropped(t *testing.T) {
	params, _, err := encodeChatRequest("gpt-4o", &koine.LanguageRequest{
		Messages: []koine.Message{
			koine.UserText("draw a cat"),
			{Role: koine.RoleAssistant, Blocks: koine.Blocks{
				&koine.TextBlock{Text: "here you go"},
				&koine.ImageBlock{MIMEType: "image/png", Data: []byte{1, 2, 3}},
			}},
			koine.UserText("now describe it"),
		},
	})
	if err != nil {
		t.Fatalf("encodeChatRequest() = %v, want assistant image dropped", err)
	}
	if len(params.Messages) != 3 {
		t.Errorf("len(Messages) = %d, want 3", len(params.Messages))
	}
}

func TestChatStreamEventsAndResponse(t *testing.T) {
	var body json.RawMessage
	server := chatServer(t, &body, chatFixture...)
	d := New(WithAPIKey("test"), WithBaseURL(server.URL)).ChatCompletionsModel("deepseek-test")

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
		koine.EventToolCallStart, koine.EventToolCallDelta, koine.EventToolCallEnd,
		koine.EventUsage, koine.EventStop,
	}
	var gotTypes []koine.EventType
	for _, e := range events {
		gotTypes = append(gotTypes, e.Type)
	}
	if !reflect.DeepEqual(gotTypes, wantTypes) {
		t.Fatalf("event types = %v, want %v", gotTypes, wantTypes)
	}
	if events[0].Index == events[1].Index || events[1].Index == events[3].Index {
		t.Errorf("block indices must advance across kinds: %v", gotTypes)
	}
	end := events[5]
	if end.ToolCall.ID != "call_1" || string(end.ToolCall.Input) != `{"city":"SF"}` {
		t.Errorf("tool_call_end = %#v", end.ToolCall)
	}
	if events[7].StopReason != koine.StopToolUse {
		t.Errorf("stop = %q", events[7].StopReason)
	}

	resp := stream.Response()
	if resp == nil {
		t.Fatal("nil response")
	}
	wantUsage := koine.Usage{InputTokens: 8, OutputTokens: 5, CacheReadTokens: 2, ReasoningTokens: 3}
	if resp.Usage != wantUsage {
		t.Errorf("usage = %#v", resp.Usage)
	}
	if resp.Model != "deepseek-test" || resp.StopReason != koine.StopToolUse {
		t.Errorf("meta = %#v", resp)
	}
	blocks := resp.Message.Blocks
	if len(blocks) != 3 {
		t.Fatalf("blocks = %#v", blocks)
	}
	if th := blocks[0].(*koine.ThinkingBlock); th.Text != "pondering" || th.Raw == nil || th.Raw.Provider != Name || string(th.Raw.JSON) != `{"reasoning_content":"pondering"}` {
		t.Errorf("thinking = %#v", th)
	}
	if tx := blocks[1].(*koine.TextBlock); tx.Text != "Hi there" {
		t.Errorf("text = %#v", tx)
	}
	if tu := blocks[2].(*koine.ToolUseBlock); tu.ID != "call_1" || string(tu.Input) != `{"city":"SF"}` {
		t.Errorf("tool = %#v", tu)
	}
}

func TestChatReasoningField(t *testing.T) {
	server := chatServer(t, new(json.RawMessage),
		chatChunk(`"choices":[{"index":0,"delta":{"role":"assistant","reasoning_content":null,"reasoning":"weigh "},"finish_reason":null}]`),
		chatChunk(`"choices":[{"index":0,"delta":{"reasoning_content":"","reasoning":"it"},"finish_reason":null}]`),
		chatChunk(`"choices":[{"index":0,"delta":{"content":"done"},"finish_reason":"stop"}]`),
	)
	stream, err := New(WithAPIKey("test"), WithBaseURL(server.URL)).ChatCompletionsModel("kimi-test").Stream(context.Background(), &koine.LanguageRequest{
		Messages: []koine.Message{koine.UserText("hi")},
	})
	if err != nil {
		t.Fatal(err)
	}
	var streamed strings.Builder
	for e := range stream.Events() {
		if e.Type == koine.EventThinkingDelta {
			streamed.WriteString(e.Text)
		}
	}
	stream.Close()
	if err := stream.Err(); err != nil {
		t.Fatal(err)
	}
	th, ok := stream.Response().Message.Blocks[0].(*koine.ThinkingBlock)
	if !ok || th.Text != "weigh it" || streamed.String() != "weigh it" || string(th.Raw.JSON) != `{"reasoning":"weigh it"}` {
		t.Errorf("thinking = %#v, streamed %q", th, streamed.String())
	}
}

func TestChatReasoningReplay(t *testing.T) {
	raw := func(provider, json string) *koine.ProviderRaw {
		return &koine.ProviderRaw{Provider: provider, JSON: []byte(json)}
	}
	cases := []struct {
		name  string
		block *koine.ThinkingBlock
		opts  LanguageOptions
		want  map[string]any
	}{
		{"reasoning_content", &koine.ThinkingBlock{Text: "R1", Raw: raw(Name, `{"reasoning_content":"R1"}`)}, LanguageOptions{}, map[string]any{"reasoning_content": "R1"}},
		{"reasoning", &koine.ThinkingBlock{Text: "R2", Raw: raw(Name, `{"reasoning":"R2"}`)}, LanguageOptions{}, map[string]any{"reasoning": "R2"}},
		{"replay off", &koine.ThinkingBlock{Text: "R1", Raw: raw(Name, `{"reasoning_content":"R1"}`)}, LanguageOptions{NoReasoningReplay: true}, map[string]any{}},
		{"responses item", &koine.ThinkingBlock{Text: "R3", Raw: raw(Name, `{"type":"reasoning","summary":[],"encrypted_content":"X"}`)}, LanguageOptions{}, map[string]any{}},
		{"other provider", &koine.ThinkingBlock{Text: "R4", Raw: raw("anthropic", `{"reasoning_content":"R4"}`)}, LanguageOptions{}, map[string]any{}},
		{"no raw", &koine.ThinkingBlock{Text: "R5"}, LanguageOptions{}, map[string]any{}},
	}
	for _, c := range cases {
		params, _, err := encodeChatRequest("deepseek-test", &koine.LanguageRequest{
			Messages: []koine.Message{
				koine.UserText("hi"),
				{Role: koine.RoleAssistant, Blocks: koine.Blocks{c.block, &koine.ToolUseBlock{ID: "call_1", Name: "f", Input: json.RawMessage(`{}`)}}},
				koine.ToolResultText("call_1", "ok", false),
			},
			ProviderOptions: map[string]any{Name: c.opts},
		})
		if err != nil {
			t.Fatalf("%s: %v", c.name, err)
		}
		data, err := json.Marshal(params.Messages[1])
		if err != nil {
			t.Fatal(err)
		}
		var msg map[string]any
		if err := json.Unmarshal(data, &msg); err != nil {
			t.Fatal(err)
		}
		for _, field := range []string{"role", "content", "tool_calls"} {
			delete(msg, field)
		}
		if !reflect.DeepEqual(msg, c.want) {
			t.Errorf("%s: extra assistant fields = %v, want %v (message %s)", c.name, msg, c.want, data)
		}
	}
}

func TestChatEncodeRequestWire(t *testing.T) {
	var body json.RawMessage
	server := chatServer(t, &body, chatFixture...)
	d := New(WithAPIKey("test"), WithBaseURL(server.URL)).ChatCompletionsModel("gpt-test")

	req := &koine.LanguageRequest{
		System:    "be brief",
		MaxTokens: 500,
		Thinking:  &koine.Thinking{Effort: koine.ThinkingLow},
		Tools: []koine.Tool{{
			Name:        "get_weather",
			Description: "weather lookup",
			InputSchema: map[string]any{
				"type":       "object",
				"properties": map[string]any{"city": map[string]any{"type": "string"}},
			},
		}},
		ToolChoice: &koine.ToolChoice{Mode: koine.ToolChoiceTool, Name: "get_weather"},
		Messages: []koine.Message{
			koine.UserText("hi"),
			{Role: koine.RoleAssistant, Blocks: koine.Blocks{
				&koine.ThinkingBlock{Text: "dropped"},
				&koine.TextBlock{Text: "checking"},
				&koine.ToolUseBlock{ID: "call_1", Name: "get_weather", Input: json.RawMessage(`{"city":"SF"}`)},
			}},
			koine.ToolResultText("call_1", "sunny", false),
		},
		ProviderOptions: map[string]any{Name: LanguageOptions{ExtraBody: map[string]any{"enable_thinking": true}}},
	}
	stream, err := d.Stream(context.Background(), req)
	if err != nil {
		t.Fatal(err)
	}
	for stream.Next() {
	}
	stream.Close()

	var wire struct {
		Model               string `json:"model"`
		MaxCompletionTokens int    `json:"max_completion_tokens"`
		MaxTokens           int    `json:"max_tokens"`
		ReasoningEffort     string `json:"reasoning_effort"`
		EnableThinking      bool   `json:"enable_thinking"`
		StreamOptions       struct {
			IncludeUsage bool `json:"include_usage"`
		} `json:"stream_options"`
		ToolChoice struct {
			Type     string `json:"type"`
			Function struct {
				Name string `json:"name"`
			} `json:"function"`
		} `json:"tool_choice"`
		Tools []struct {
			Type     string `json:"type"`
			Function struct {
				Name       string         `json:"name"`
				Parameters map[string]any `json:"parameters"`
			} `json:"function"`
		} `json:"tools"`
		Messages []struct {
			Role       string `json:"role"`
			Content    any    `json:"content"`
			ToolCallID string `json:"tool_call_id"`
			ToolCalls  []struct {
				ID       string `json:"id"`
				Function struct {
					Name      string `json:"name"`
					Arguments string `json:"arguments"`
				} `json:"function"`
			} `json:"tool_calls"`
		} `json:"messages"`
	}
	if err := json.Unmarshal(body, &wire); err != nil {
		t.Fatalf("decode wire: %v\n%s", err, body)
	}
	if wire.Model != "gpt-test" || wire.MaxCompletionTokens != 500 || wire.MaxTokens != 0 {
		t.Errorf("model/max: %+v", wire)
	}
	if wire.ReasoningEffort != "low" || !wire.EnableThinking || !wire.StreamOptions.IncludeUsage {
		t.Errorf("thinking/extra/stream: %+v", wire)
	}
	if wire.ToolChoice.Function.Name != "get_weather" {
		t.Errorf("tool_choice: %+v", wire.ToolChoice)
	}
	if wire.Tools[0].Function.Parameters["type"] != "object" {
		t.Errorf("tool schema: %+v", wire.Tools[0])
	}
	if len(wire.Messages) != 4 {
		t.Fatalf("messages: %d\n%s", len(wire.Messages), body)
	}
	if wire.Messages[0].Role != "system" || wire.Messages[1].Role != "user" {
		t.Errorf("roles: %+v", wire.Messages)
	}
	assistant := wire.Messages[2]
	if assistant.Content != "checking" || len(assistant.ToolCalls) != 1 || assistant.ToolCalls[0].Function.Arguments != `{"city":"SF"}` {
		t.Errorf("assistant: %+v", assistant)
	}
	toolMsg := wire.Messages[3]
	if toolMsg.Role != "tool" || toolMsg.ToolCallID != "call_1" || toolMsg.Content != "sunny" {
		t.Errorf("tool message: %+v", toolMsg)
	}
}

func TestChatLegacyMaxTokens(t *testing.T) {
	var body json.RawMessage
	server := chatServer(t, &body, chatFixture...)
	d := New(WithAPIKey("test"), WithBaseURL(server.URL)).ChatCompletionsModel("old-compat")

	req := &koine.LanguageRequest{
		MaxTokens:       300,
		Messages:        []koine.Message{koine.UserText("hi")},
		ProviderOptions: map[string]any{Name: LanguageOptions{LegacyMaxTokens: true, NoStreamUsage: true}},
	}
	stream, err := d.Stream(context.Background(), req)
	if err != nil {
		t.Fatal(err)
	}
	for stream.Next() {
	}
	stream.Close()

	var wire map[string]json.RawMessage
	if err := json.Unmarshal(body, &wire); err != nil {
		t.Fatal(err)
	}
	if string(wire["max_tokens"]) != "300" {
		t.Errorf("max_tokens = %s", wire["max_tokens"])
	}
	if _, ok := wire["max_completion_tokens"]; ok {
		t.Error("max_completion_tokens must be absent with LegacyMaxTokens")
	}
	if _, ok := wire["stream_options"]; ok {
		t.Error("stream_options must be absent with NoStreamUsage")
	}
}

func TestChatDecodeUsageCacheExceedsPrompt(t *testing.T) {
	usage, ok := decodeChatUsage(openai.CompletionUsage{
		PromptTokens:        50,
		PromptTokensDetails: openai.CompletionUsagePromptTokensDetails{CachedTokens: 12000},
	})
	if !ok {
		t.Fatal("decodeChatUsage() ok = false, want true")
	}
	if usage.InputTokens != 0 || usage.CacheReadTokens != 12000 {
		t.Errorf("usage = %#v, want InputTokens 0 and CacheReadTokens 12000", usage)
	}
}

func TestChatDecodeUsageCacheWrite(t *testing.T) {
	usage, _ := decodeChatUsage(openai.CompletionUsage{
		PromptTokens:        12625,
		PromptTokensDetails: openai.CompletionUsagePromptTokensDetails{CacheWriteTokens: 12622},
	})
	if usage.InputTokens != 3 || usage.CacheWriteTokens != 12622 {
		t.Errorf("usage = %#v, want InputTokens 3 and CacheWriteTokens 12622", usage)
	}
}

func TestChatErrorMapping(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusUnauthorized)
		fmt.Fprint(w, `{"error":{"message":"bad key","type":"invalid_request_error","code":"invalid_api_key"}}`)
	}))
	defer server.Close()
	d := New(WithAPIKey("test"), WithBaseURL(server.URL)).ChatCompletionsModel("gpt-test")

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
	if kerr.Status != http.StatusUnauthorized || kerr.Retryable() {
		t.Errorf("error = %#v", kerr)
	}
	if kerr.Message != "bad key" {
		t.Errorf("message = %q, want clean provider message", kerr.Message)
	}
}

func TestChatThinkingWire(t *testing.T) {
	cases := []struct {
		thinking *koine.Thinking
		want     string
	}{
		{&koine.Thinking{Effort: koine.ThinkingNone}, "none"},
		{&koine.Thinking{BudgetTokens: 8000}, "medium"},
		{&koine.Thinking{Effort: koine.ThinkingLow, BudgetTokens: 8000}, "low"},
	}
	for _, c := range cases {
		var body json.RawMessage
		server := chatServer(t, &body, chatFixture...)
		d := New(WithAPIKey("test"), WithBaseURL(server.URL)).ChatCompletionsModel("gpt-test")
		stream, err := d.Stream(context.Background(), &koine.LanguageRequest{
			Thinking: c.thinking,
			Messages: []koine.Message{koine.UserText("hi")},
		})
		if err != nil {
			t.Fatal(err)
		}
		for stream.Next() {
		}
		stream.Close()

		var wire struct {
			ReasoningEffort string `json:"reasoning_effort"`
		}
		if err := json.Unmarshal(body, &wire); err != nil {
			t.Fatalf("decode wire: %v\n%s", err, body)
		}
		if wire.ReasoningEffort != c.want {
			t.Errorf("%+v: reasoning_effort = %q, want %q", c.thinking, wire.ReasoningEffort, c.want)
		}
	}

	server := chatServer(t, new(json.RawMessage), chatFixture...)
	d := New(WithAPIKey("test"), WithBaseURL(server.URL)).ChatCompletionsModel("gpt-test")
	if _, err := d.Stream(context.Background(), &koine.LanguageRequest{
		Thinking: &koine.Thinking{Effort: koine.ThinkingNone, BudgetTokens: 100},
		Messages: []koine.Message{koine.UserText("hi")},
	}); err == nil {
		t.Fatal("ThinkingNone with BudgetTokens: want error")
	}
}

func TestChatResponseFormatWire(t *testing.T) {
	var body json.RawMessage
	server := chatServer(t, &body, chatFixture...)
	d := New(WithAPIKey("test"), WithBaseURL(server.URL)).ChatCompletionsModel("gpt-test")

	stream, err := d.Stream(context.Background(), &koine.LanguageRequest{
		Messages: []koine.Message{koine.UserText("hi")},
		ResponseFormat: &koine.ResponseFormat{
			Schema:      map[string]any{"type": "object", "properties": map[string]any{"answer": map[string]any{"type": "string"}}},
			Description: "the answer",
			Strict:      true,
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	for stream.Next() {
	}
	stream.Close()

	var wire struct {
		ResponseFormat struct {
			Type       string `json:"type"`
			JSONSchema struct {
				Name        string         `json:"name"`
				Description string         `json:"description"`
				Strict      bool           `json:"strict"`
				Schema      map[string]any `json:"schema"`
			} `json:"json_schema"`
		} `json:"response_format"`
	}
	if err := json.Unmarshal(body, &wire); err != nil {
		t.Fatal(err)
	}
	rf := wire.ResponseFormat
	if rf.Type != "json_schema" || rf.JSONSchema.Name != "response" || !rf.JSONSchema.Strict || rf.JSONSchema.Description != "the answer" || rf.JSONSchema.Schema["type"] != "object" {
		t.Errorf("response_format = %+v", rf)
	}
}

func TestChatResponseFormatSchemaless(t *testing.T) {
	var body json.RawMessage
	server := chatServer(t, &body, chatFixture...)
	d := New(WithAPIKey("test"), WithBaseURL(server.URL)).ChatCompletionsModel("gpt-test")

	stream, err := d.Stream(context.Background(), &koine.LanguageRequest{
		Messages:       []koine.Message{koine.UserText("hi")},
		ResponseFormat: &koine.ResponseFormat{},
	})
	if err != nil {
		t.Fatal(err)
	}
	for stream.Next() {
	}
	stream.Close()

	var wire struct {
		ResponseFormat struct {
			Type string `json:"type"`
		} `json:"response_format"`
	}
	if err := json.Unmarshal(body, &wire); err != nil {
		t.Fatal(err)
	}
	if wire.ResponseFormat.Type != "json_object" {
		t.Errorf("response_format = %+v", wire.ResponseFormat)
	}
}

func TestChatToolResultJSONWire(t *testing.T) {
	var body json.RawMessage
	server := chatServer(t, &body, chatFixture...)
	d := New(WithAPIKey("test"), WithBaseURL(server.URL)).ChatCompletionsModel("gpt-test")

	stream, err := d.Stream(context.Background(), &koine.LanguageRequest{
		Messages: []koine.Message{
			koine.UserText("hi"),
			{Role: koine.RoleAssistant, Blocks: koine.Blocks{
				&koine.ToolUseBlock{ID: "call_1", Name: "get_weather", Input: json.RawMessage(`{}`)},
			}},
			koine.ToolResultJSON("call_1", json.RawMessage(`{"temp":22}`), false),
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
			Role    string `json:"role"`
			Content any    `json:"content"`
		} `json:"messages"`
	}
	if err := json.Unmarshal(body, &wire); err != nil {
		t.Fatal(err)
	}
	last := wire.Messages[len(wire.Messages)-1]
	if last.Role != "tool" || last.Content != `{"temp":22}` {
		t.Errorf("tool message = %+v", last)
	}
}
