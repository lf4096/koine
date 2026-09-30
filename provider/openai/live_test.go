package openai

import (
	"context"
	"encoding/json"
	"testing"

	"github.com/lf4096/koine"
	"github.com/lf4096/koine/internal/testenv"
)

func TestLiveChatStream(t *testing.T) {
	d := New(liveOpts(t)...).ChatCompletionsModel(testenv.Get("KOINE_TEST_OPENAI_CHAT_MODEL", "gpt-4.1-mini"))

	stream, err := d.Stream(context.Background(), &koine.LanguageRequest{
		MaxTokens: 64,
		Messages:  []koine.Message{koine.UserText("Reply with one word: pong")},
	})
	if err != nil {
		t.Fatal(err)
	}
	defer stream.Close()
	sawText := false
	for stream.Next() {
		if stream.Event().Type == koine.EventTextDelta {
			sawText = true
		}
	}
	if err := stream.Err(); err != nil {
		t.Fatal(err)
	}
	resp := stream.Response()
	if !sawText || resp.Message.Text() == "" {
		t.Fatalf("no text streamed: %#v", resp)
	}
	if resp.Usage.OutputTokens == 0 {
		t.Errorf("usage not populated: %#v", resp.Usage)
	}
}

func TestLiveToolRoundTrip(t *testing.T) {
	m := New(liveOpts(t)...).LanguageModel(testenv.Key(t, "KOINE_TEST_OPENAI_MODEL"))
	ctx := context.Background()

	req := &koine.LanguageRequest{
		MaxTokens: 2048,
		Thinking:  &koine.Thinking{Effort: koine.ThinkingHigh},
		Tools: []koine.Tool{{
			Name:        "get_weather",
			Description: "Get current weather for a city",
			InputSchema: map[string]any{
				"type":       "object",
				"properties": map[string]any{"city": map[string]any{"type": "string"}},
				"required":   []string{"city"},
			},
		}},
		Messages: []koine.Message{koine.UserText("What's the weather in Paris? Use the tool.")},
	}
	resp, err := m.Complete(ctx, req)
	if err != nil {
		t.Fatal(err)
	}
	if resp.StopReason != koine.StopToolUse || len(resp.Message.ToolUses()) == 0 {
		t.Fatalf("expected tool use, got %q: %#v", resp.StopReason, resp.Message)
	}
	if resp.Usage.InputTokens == 0 || resp.Usage.OutputTokens == 0 {
		t.Errorf("usage not populated: %#v", resp.Usage)
	}
	// Only OpenAI returns encrypted reasoning, and it accepts a replay without
	// it, so nothing but this check notices when it goes missing.
	openAI := testenv.Get("KOINE_TEST_OPENAI_BASE_URL", "") == ""
	reasoned := false
	for _, b := range resp.Message.Blocks {
		if th, ok := b.(*koine.ThinkingBlock); ok {
			reasoned = true
			if enc, _ := rawField(t, th.Raw, "encrypted_content").(string); openAI && enc == "" {
				t.Fatalf("reasoning item without encrypted_content: %s", th.Raw.JSON)
			}
		}
	}

	call := resp.Message.ToolUses()[0]
	req.Messages = append(req.Messages, resp.Message, koine.ToolResultJSON(call.ID, json.RawMessage(`{"temperature":"22C","condition":"sunny"}`), false))
	resp2, err := m.Complete(ctx, req)
	if err != nil {
		t.Fatal(err)
	}
	if resp2.StopReason != koine.StopEndTurn || resp2.Message.Text() == "" {
		t.Errorf("empty final answer, stop %q: %#v", resp2.StopReason, resp2.Message)
	}
	if !reasoned {
		t.Skip("turn 1 returned no reasoning item; reasoning replay not exercised")
	}
}
