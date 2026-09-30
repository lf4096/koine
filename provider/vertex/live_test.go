package vertex

import (
	"context"
	"slices"
	"testing"

	"github.com/lf4096/koine"
	"github.com/lf4096/koine/internal/testenv"
)

// Live tests target a Vertex-style gateway (bearer auth + custom base URL);
// they skip without KOINE_TEST_VERTEX_BEARER.
func liveOpts(t *testing.T) []Option {
	t.Helper()
	bearer := testenv.Key(t, "KOINE_TEST_VERTEX_BEARER")
	opts := []Option{WithHeader("Authorization", "Bearer "+bearer)}
	if base := testenv.Get("KOINE_TEST_VERTEX_BASE_URL", ""); base != "" {
		opts = append(opts, WithBaseURL(base))
	}
	if v := testenv.Get("KOINE_TEST_VERTEX_API_VERSION", ""); v != "" {
		opts = append(opts, WithAPIVersion(v))
	}
	return opts
}

func TestLiveEmbed(t *testing.T) {
	p, err := New(context.Background(), liveOpts(t)...)
	if err != nil {
		t.Fatal(err)
	}
	m := p.EmbeddingModel(testenv.Get("KOINE_TEST_VERTEX_EMBED_MODEL", "gemini-embedding-001"))
	resp, err := m.Embed(context.Background(), &koine.EmbedRequest{
		Inputs: []string{"the common tongue", "one tongue for every LLM"},
		Task:   koine.EmbedTaskDocument,
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(resp.Embeddings) != 2 || len(resp.Embeddings[0]) == 0 {
		t.Fatalf("embeddings shape: %d x %d", len(resp.Embeddings), len(resp.Embeddings[0]))
	}
}

func TestLiveImageOutput(t *testing.T) {
	p, err := New(context.Background(), liveOpts(t)...)
	if err != nil {
		t.Fatal(err)
	}
	m := p.LanguageModel(testenv.Get("KOINE_TEST_VERTEX_IMAGE_MODEL", "gemini-2.5-flash-image"))
	req := &koine.LanguageRequest{Messages: []koine.Message{koine.UserText("Draw a small red circle on a white background.")}}
	resp, err := m.Complete(context.Background(), req)
	if err != nil {
		t.Fatal(err)
	}
	if !slices.ContainsFunc(resp.Message.Blocks, func(b koine.Block) bool {
		img, ok := b.(*koine.ImageBlock)
		return ok && len(img.Data) > 0
	}) {
		t.Fatalf("no image in %#v", resp.Message.Blocks)
	}
	// Gemini 3 image models reject this turn unless the image's thought
	// signature comes back with it.
	req.Messages = append(req.Messages, resp.Message, koine.UserText("Now make the circle blue."))
	if _, err := m.Complete(context.Background(), req); err != nil {
		t.Fatal(err)
	}
}
