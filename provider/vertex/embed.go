package vertex

import (
	"context"

	"github.com/lf4096/koine"
	"github.com/lf4096/koine/internal/genaiengine"
)

// EmbedOptions carries Vertex-specific embedding parameters.
type EmbedOptions = genaiengine.EmbedOptions

// EmbeddingModel embeds text via the Vertex predict endpoint.
type EmbeddingModel struct {
	e genaiengine.Engine
}

// NewEmbeddingModel builds the embedding model; options as NewLanguageModel.
func NewEmbeddingModel(ctx context.Context, opts ...Option) (*EmbeddingModel, error) {
	e, err := newEngine(ctx, opts)
	if err != nil {
		return nil, err
	}
	return &EmbeddingModel{e: e}, nil
}

func (m *EmbeddingModel) Name() string { return Name }

// Embed performs one embedding call per input (the Vertex endpoint takes one
// content at a time).
func (m *EmbeddingModel) Embed(ctx context.Context, req *koine.EmbedRequest) (*koine.EmbedResponse, error) {
	return m.e.Embed(ctx, req)
}
