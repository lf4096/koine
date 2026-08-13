package anthropic

import (
	"context"
	"encoding/json"
	"testing"

	"github.com/lf4096/koine"
	"github.com/lf4096/koine/internal/testenv"
)

func TestLiveToolRoundTrip(t *testing.T) {
	key := testenv.Key(t, "KOINE_TEST_ANTHROPIC_API_KEY")
	model := testenv.Get("KOINE_TEST_ANTHROPIC_MODEL", "claude-sonnet-4-5")
	opts := []Option{WithAPIKey(key)}
	if base := testenv.Get("KOINE_TEST_ANTHROPIC_BASE_URL", ""); base != "" {
		opts = append(opts, WithBaseURL(base))
	}
	d := New(opts...).LanguageModel(model)
	ctx := context.Background()

	req := &koine.LanguageRequest{
		MaxTokens: 256,
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
	resp, err := d.Complete(ctx, req)
	if err != nil {
		t.Fatal(err)
	}
	if resp.StopReason != koine.StopToolUse || len(resp.Message.ToolUses()) == 0 {
		t.Fatalf("expected tool use, got %q: %#v", resp.StopReason, resp.Message)
	}
	if resp.Usage.InputTokens == 0 || resp.Usage.OutputTokens == 0 {
		t.Errorf("usage not populated: %#v", resp.Usage)
	}

	call := resp.Message.ToolUses()[0]
	req.Messages = append(req.Messages, resp.Message, koine.ToolResultJSON(call.ID, json.RawMessage(`{"temperature":"22C","condition":"sunny"}`), false))
	resp2, err := d.Complete(ctx, req)
	if err != nil {
		t.Fatal(err)
	}
	if resp2.Message.Text() == "" {
		t.Errorf("empty final answer: %#v", resp2.Message)
	}
}
