package openai

import (
	"context"
	"testing"

	"github.com/lf4096/koine"
	"github.com/lf4096/koine/internal/testenv"
)

func TestLiveStream(t *testing.T) {
	key := testenv.Key(t, "KOINE_TEST_OPENAI_API_KEY")
	model := testenv.Get("KOINE_TEST_OPENAI_MODEL", "gpt-4.1-mini")
	opts := []Option{WithAPIKey(key)}
	if base := testenv.Get("KOINE_TEST_OPENAI_BASE_URL", ""); base != "" {
		opts = append(opts, WithBaseURL(base))
	}
	d := New(opts...).LanguageModel(model)

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
