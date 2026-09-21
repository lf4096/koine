package gemini

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

	"google.golang.org/genai"

	"github.com/lf4096/koine"
)

func sseServer(t *testing.T, capture *json.RawMessage, chunks ...string) *httptest.Server {
	t.Helper()
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body json.RawMessage
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Errorf("decode request body: %v", err)
		}
		*capture = body
		w.Header().Set("Content-Type", "text/event-stream")
		for _, c := range chunks {
			fmt.Fprintf(w, "data: %s\r\n\r\n", c)
		}
	}))
	t.Cleanup(server.Close)
	return server
}

func newTestModel(t *testing.T, serverURL string) *LanguageModel {
	t.Helper()
	p, err := New(context.Background(), WithAPIKey("test"), WithBaseURL(serverURL))
	if err != nil {
		t.Fatal(err)
	}
	return p.LanguageModel("gemini-test")
}

// "c2ln" is base64("sig"): ThoughtSignature is []byte and travels base64 in JSON.
var streamFixture = []string{
	`{"candidates":[{"content":{"role":"model","parts":[{"text":"pondering","thought":true}]},"index":0}],"modelVersion":"gemini-test"}`,
	`{"candidates":[{"content":{"role":"model","parts":[{"text":" deeply","thought":true,"thoughtSignature":"c2ln"}]},"index":0}]}`,
	`{"candidates":[{"content":{"role":"model","parts":[{"text":"Hello"}]},"index":0}]}`,
	`{"candidates":[{"content":{"role":"model","parts":[{"functionCall":{"name":"get_weather","args":{"city":"SF"}}}]},"finishReason":"STOP","index":0}],"usageMetadata":{"promptTokenCount":7,"candidatesTokenCount":4,"thoughtsTokenCount":6,"cachedContentTokenCount":1}}`,
}

func TestStreamEventsAndResponse(t *testing.T) {
	var body json.RawMessage
	server := sseServer(t, &body, streamFixture...)
	d := newTestModel(t, server.URL)

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
		koine.EventThinkingDelta, koine.EventThinkingDelta,
		koine.EventTextDelta,
		koine.EventToolCallStart, koine.EventToolCallEnd,
		koine.EventUsage, koine.EventStop,
	}
	var gotTypes []koine.EventType
	for _, e := range events {
		gotTypes = append(gotTypes, e.Type)
	}
	if !reflect.DeepEqual(gotTypes, wantTypes) {
		t.Fatalf("event types = %v, want %v", gotTypes, wantTypes)
	}
	end := events[4]
	if end.ToolCall.ID != "koine:2" || end.ToolCall.Name != "get_weather" || string(end.ToolCall.Input) != `{"city":"SF"}` {
		t.Errorf("tool_call_end = %#v", end.ToolCall)
	}
	if events[6].StopReason != koine.StopToolUse {
		t.Errorf("stop = %q", events[6].StopReason)
	}

	resp := stream.Response()
	if resp == nil {
		t.Fatal("nil response")
	}
	wantUsage := koine.Usage{InputTokens: 6, OutputTokens: 10, CacheReadTokens: 1, ReasoningTokens: 6}
	if resp.Usage != wantUsage {
		t.Errorf("usage = %#v", resp.Usage)
	}
	if resp.Model != "gemini-test" || resp.Provider != Name {
		t.Errorf("meta = %#v", resp)
	}
	blocks := resp.Message.Blocks
	if len(blocks) != 3 {
		t.Fatalf("blocks = %#v", blocks)
	}
	thinking := blocks[0].(*koine.ThinkingBlock)
	if thinking.Text != "pondering deeply" || thinking.Signature != "c2ln" {
		t.Errorf("thinking = %#v", thinking)
	}
	if thinking.Raw == nil || !strings.Contains(string(thinking.Raw.JSON), `"thoughtSignature":"c2ln"`) {
		t.Errorf("thinking raw = %#v", thinking.Raw)
	}
	if tx := blocks[1].(*koine.TextBlock); tx.Text != "Hello" {
		t.Errorf("text = %#v", tx)
	}
	tool := blocks[2].(*koine.ToolUseBlock)
	if tool.ID != "koine:2" || tool.Raw == nil {
		t.Errorf("tool = %#v", tool)
	}
}

func TestEncodeRequestWire(t *testing.T) {
	var body json.RawMessage
	server := sseServer(t, &body, streamFixture...)
	d := newTestModel(t, server.URL)

	req := &koine.LanguageRequest{
		System:    "be brief",
		MaxTokens: 800,
		Thinking:  &koine.Thinking{BudgetTokens: 2048},
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
				&koine.ToolUseBlock{ID: "koine:0", Name: "get_weather", Input: json.RawMessage(`{"city":"SF"}`)},
			}},
			koine.ToolResultText("koine:0", "sunny", false),
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
		SystemInstruction struct {
			Parts []struct {
				Text string `json:"text"`
			} `json:"parts"`
		} `json:"systemInstruction"`
		GenerationConfig struct {
			MaxOutputTokens int `json:"maxOutputTokens"`
			ThinkingConfig  struct {
				IncludeThoughts bool `json:"includeThoughts"`
				ThinkingBudget  int  `json:"thinkingBudget"`
			} `json:"thinkingConfig"`
		} `json:"generationConfig"`
		Tools []struct {
			FunctionDeclarations []struct {
				Name                 string         `json:"name"`
				ParametersJsonSchema map[string]any `json:"parametersJsonSchema"`
			} `json:"functionDeclarations"`
		} `json:"tools"`
		ToolConfig struct {
			FunctionCallingConfig struct {
				Mode                 string   `json:"mode"`
				AllowedFunctionNames []string `json:"allowedFunctionNames"`
			} `json:"functionCallingConfig"`
		} `json:"toolConfig"`
		Contents []struct {
			Role  string `json:"role"`
			Parts []struct {
				Text         string `json:"text"`
				FunctionCall *struct {
					ID   string         `json:"id"`
					Name string         `json:"name"`
					Args map[string]any `json:"args"`
				} `json:"functionCall"`
				FunctionResponse *struct {
					ID       string         `json:"id"`
					Name     string         `json:"name"`
					Response map[string]any `json:"response"`
				} `json:"functionResponse"`
			} `json:"parts"`
		} `json:"contents"`
	}
	if err := json.Unmarshal(body, &wire); err != nil {
		t.Fatalf("decode wire: %v\n%s", err, body)
	}
	if wire.SystemInstruction.Parts[0].Text != "be brief" {
		t.Errorf("system: %+v", wire.SystemInstruction)
	}
	gc := wire.GenerationConfig
	if gc.MaxOutputTokens != 800 || !gc.ThinkingConfig.IncludeThoughts || gc.ThinkingConfig.ThinkingBudget != 2048 {
		t.Errorf("generationConfig: %+v", gc)
	}
	fd := wire.Tools[0].FunctionDeclarations[0]
	if fd.Name != "get_weather" || fd.ParametersJsonSchema["type"] != "object" {
		t.Errorf("function declaration: %+v", fd)
	}
	fcc := wire.ToolConfig.FunctionCallingConfig
	if fcc.Mode != "ANY" || !reflect.DeepEqual(fcc.AllowedFunctionNames, []string{"get_weather"}) {
		t.Errorf("toolConfig: %+v", fcc)
	}
	if len(wire.Contents) != 3 {
		t.Fatalf("contents: %d\n%s", len(wire.Contents), body)
	}
	model := wire.Contents[1]
	if model.Role != "model" || model.Parts[0].FunctionCall == nil {
		t.Fatalf("model turn: %+v", model)
	}
	// Synthetic ids never reach the wire.
	if model.Parts[0].FunctionCall.ID != "" || model.Parts[0].FunctionCall.Args["city"] != "SF" {
		t.Errorf("functionCall: %+v", model.Parts[0].FunctionCall)
	}
	toolTurn := wire.Contents[2]
	fr := toolTurn.Parts[0].FunctionResponse
	if toolTurn.Role != "user" || fr == nil || fr.ID != "" || fr.Name != "get_weather" || fr.Response["output"] != "sunny" {
		t.Errorf("functionResponse: %+v", fr)
	}
}

func TestThinkingRawRoundTrip(t *testing.T) {
	var body json.RawMessage
	server := sseServer(t, &body, streamFixture...)
	d := newTestModel(t, server.URL)

	rawJSON := `{"text":"pondering","thought":true,"thoughtSignature":"c2ln"}`
	req := &koine.LanguageRequest{
		Messages: []koine.Message{
			koine.UserText("hi"),
			{Role: koine.RoleAssistant, Blocks: koine.Blocks{
				&koine.ThinkingBlock{Text: "pondering", Signature: "c2ln", Raw: &koine.ProviderRaw{Provider: Name, JSON: json.RawMessage(rawJSON)}},
				&koine.TextBlock{Text: "Hello"},
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
		Contents []struct {
			Parts []map[string]any `json:"parts"`
		} `json:"contents"`
	}
	if err := json.Unmarshal(body, &wire); err != nil {
		t.Fatal(err)
	}
	parts := wire.Contents[1].Parts
	if len(parts) != 2 {
		t.Fatalf("model parts: %+v", parts)
	}
	if parts[0]["thought"] != true || parts[0]["thoughtSignature"] != "c2ln" {
		t.Errorf("thought part resend = %+v", parts[0])
	}
}

func TestToolResultWithoutMatchingCall(t *testing.T) {
	var body json.RawMessage
	server := sseServer(t, &body, streamFixture...)
	d := newTestModel(t, server.URL)

	_, err := d.Stream(context.Background(), &koine.LanguageRequest{
		Messages: []koine.Message{koine.ToolResultText("orphan", "x", false)},
	})
	if err == nil || !strings.Contains(err.Error(), "no matching tool call") {
		t.Errorf("want missing tool call error, got %v", err)
	}
}

func TestErrorMapping(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusBadRequest)
		fmt.Fprint(w, `{"error":{"code":400,"message":"invalid argument","status":"INVALID_ARGUMENT"}}`)
	}))
	defer server.Close()
	d := newTestModel(t, server.URL)

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
	if kerr.Status != http.StatusBadRequest || kerr.Code != "INVALID_ARGUMENT" || kerr.Retryable() {
		t.Errorf("error = %#v", kerr)
	}
	if kerr.Message != "invalid argument" {
		t.Errorf("message = %q, want clean provider message", kerr.Message)
	}
}

func TestThinkingWire(t *testing.T) {
	budget := int32(128)
	cases := []struct {
		name     string
		thinking *koine.Thinking
		opts     map[string]any
		want     string
	}{
		{"effort maps to level", &koine.Thinking{Effort: koine.ThinkingHigh}, nil,
			`{"includeThoughts":true,"thinkingLevel":"HIGH"}`},
		{"none disables", &koine.Thinking{Effort: koine.ThinkingNone}, nil,
			`{"thinkingBudget":0}`},
		{"budget wins over effort", &koine.Thinking{Effort: koine.ThinkingHigh, BudgetTokens: 2048}, nil,
			`{"includeThoughts":true,"thinkingBudget":2048}`},
		{"provider options override", &koine.Thinking{Effort: koine.ThinkingHigh},
			map[string]any{Name: LanguageOptions{ThinkingConfig: &genai.ThinkingConfig{ThinkingBudget: &budget}}},
			`{"thinkingBudget":128}`},
	}
	for _, c := range cases {
		var body json.RawMessage
		server := sseServer(t, &body, streamFixture...)
		d := newTestModel(t, server.URL)
		stream, err := d.Stream(context.Background(), &koine.LanguageRequest{
			Thinking:        c.thinking,
			ProviderOptions: c.opts,
			Messages:        []koine.Message{koine.UserText("hi")},
		})
		if err != nil {
			t.Fatal(err)
		}
		for stream.Next() {
		}
		stream.Close()

		var wire struct {
			GenerationConfig struct {
				ThinkingConfig json.RawMessage `json:"thinkingConfig"`
			} `json:"generationConfig"`
		}
		if err := json.Unmarshal(body, &wire); err != nil {
			t.Fatalf("%s: decode wire: %v\n%s", c.name, err, body)
		}
		if string(wire.GenerationConfig.ThinkingConfig) != c.want {
			t.Errorf("%s: thinkingConfig = %s, want %s", c.name, wire.GenerationConfig.ThinkingConfig, c.want)
		}
	}

	server := sseServer(t, new(json.RawMessage), streamFixture...)
	d := newTestModel(t, server.URL)
	if _, err := d.Stream(context.Background(), &koine.LanguageRequest{
		Thinking: &koine.Thinking{Effort: koine.ThinkingNone, BudgetTokens: 100},
		Messages: []koine.Message{koine.UserText("hi")},
	}); err == nil {
		t.Fatal("ThinkingNone with BudgetTokens: want error")
	}
}
