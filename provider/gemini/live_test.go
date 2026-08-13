package gemini

import (
	"context"
	"encoding/json"
	"testing"

	"github.com/lf4096/koine"
	"github.com/lf4096/koine/internal/testenv"
)

func TestLiveComplete(t *testing.T) {
	key := testenv.Key(t, "KOINE_TEST_GEMINI_API_KEY")
	model := testenv.Get("KOINE_TEST_GEMINI_MODEL", "gemini-2.5-flash")
	opts := []Option{WithAPIKey(key)}
	if base := testenv.Get("KOINE_TEST_GEMINI_BASE_URL", ""); base != "" {
		opts = append(opts, WithBaseURL(base))
	}
	ctx := context.Background()
	p, err := New(ctx, opts...)
	if err != nil {
		t.Fatal(err)
	}
	d := p.LanguageModel(model)

	// Gemini thinking models spend MaxTokens on thoughts too; leave headroom.
	resp, err := d.Complete(ctx, &koine.LanguageRequest{
		MaxTokens: 512,
		Messages:  []koine.Message{koine.UserText("Reply with one word: pong")},
	})
	if err != nil {
		t.Fatal(err)
	}
	if resp.Message.Text() == "" || resp.StopReason != koine.StopEndTurn {
		t.Fatalf("unexpected response: %#v", resp)
	}
	if resp.Usage.InputTokens == 0 {
		t.Errorf("usage not populated: %#v", resp.Usage)
	}
}

func TestLiveStructuredOutput(t *testing.T) {
	key := testenv.Key(t, "KOINE_TEST_GEMINI_API_KEY")
	model := testenv.Get("KOINE_TEST_GEMINI_MODEL", "gemini-2.5-flash")
	ctx := context.Background()
	opts := []Option{WithAPIKey(key)}
	if base := testenv.Get("KOINE_TEST_GEMINI_BASE_URL", ""); base != "" {
		opts = append(opts, WithBaseURL(base))
	}
	p, err := New(ctx, opts...)
	if err != nil {
		t.Fatal(err)
	}
	d := p.LanguageModel(model)
	resp, err := d.Complete(ctx, &koine.LanguageRequest{
		MaxTokens: 512,
		Messages:  []koine.Message{koine.UserText("What is the capital of France?")},
		ResponseFormat: &koine.ResponseFormat{
			Schema: map[string]any{
				"type":       "object",
				"properties": map[string]any{"capital": map[string]any{"type": "string"}},
				"required":   []string{"capital"},
			},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	var out struct {
		Capital string `json:"capital"`
	}
	if err := json.Unmarshal([]byte(resp.Message.Text()), &out); err != nil {
		t.Fatalf("output is not schema JSON: %v\n%s", err, resp.Message.Text())
	}
	if out.Capital == "" {
		t.Fatalf("empty capital: %s", resp.Message.Text())
	}
}
