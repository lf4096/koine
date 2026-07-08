package vertex

import (
	"context"
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
	m, err := NewEmbeddingModel(context.Background(), liveOpts(t)...)
	if err != nil {
		t.Fatal(err)
	}
	resp, err := m.Embed(context.Background(), &koine.EmbedRequest{
		Model:  testenv.Get("KOINE_TEST_VERTEX_EMBED_MODEL", "gemini-embedding-001"),
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

func TestLiveGenerateImage(t *testing.T) {
	m, err := NewImageModel(context.Background(), liveOpts(t)...)
	if err != nil {
		t.Fatal(err)
	}
	resp, err := m.GenerateImage(context.Background(), &koine.ImageRequest{
		Model:       testenv.Get("KOINE_TEST_VERTEX_IMAGE_MODEL", "imagen-4.0-fast-generate-001"),
		Prompt:      "a minimalist line drawing of an ancient greek scroll",
		AspectRatio: "1:1",
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(resp.Results) == 0 || resp.Results[0].Image == nil || len(resp.Results[0].Image.Data) == 0 {
		t.Fatalf("results = %+v", resp.Results)
	}
}
