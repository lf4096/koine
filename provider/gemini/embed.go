package gemini

import (
	"context"

	"github.com/lf4096/koine"
	"github.com/lf4096/koine/internal/genaiengine"
)

// EmbedOptions carries Gemini-specific embedding parameters.
type EmbedOptions = genaiengine.EmbedOptions

// EmbeddingModel embeds text via the Gemini embedContent endpoint.
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

// Embed performs one embedContent call.
func (m *EmbeddingModel) Embed(ctx context.Context, req *koine.EmbedRequest) (*koine.EmbedResponse, error) {
	return m.e.Embed(ctx, req)
}
