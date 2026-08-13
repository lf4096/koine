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

var _ koine.EmbeddingModel = (*EmbeddingModel)(nil)

// EmbeddingModel builds the embedding model.
func (p *Provider) EmbeddingModel(model string) *EmbeddingModel {
	return &EmbeddingModel{e: p.engine(model)}
}

func (m *EmbeddingModel) Model() string { return m.e.Model }

func (m *EmbeddingModel) Provider() string { return Name }

// Embed performs one embedContent call.
func (m *EmbeddingModel) Embed(ctx context.Context, req *koine.EmbedRequest) (*koine.EmbedResponse, error) {
	return m.e.Embed(ctx, req)
}
