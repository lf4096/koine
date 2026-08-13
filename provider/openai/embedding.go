package openai

import (
	"context"

	"github.com/openai/openai-go/v3"

	"github.com/lf4096/koine"
)

// EmbedOptions carries OpenAI-specific embedding parameters. Pass via
// EmbedRequest.ProviderOptions[openai.Name].
type EmbedOptions struct {
	// User is the end-user id for abuse monitoring.
	User string
}

// EmbeddingModel embeds text via the OpenAI embeddings endpoint.
type EmbeddingModel struct {
	model  string
	client openai.Client
}

var _ koine.EmbeddingModel = (*EmbeddingModel)(nil)

// EmbeddingModel builds the embedding model.
func (p *Provider) EmbeddingModel(model string) *EmbeddingModel {
	return &EmbeddingModel{model: model, client: p.client}
}

func (m *EmbeddingModel) Model() string { return m.model }

func (m *EmbeddingModel) Provider() string { return Name }

// Embed performs one embeddings call. EmbedRequest.Task has no OpenAI
// equivalent and is ignored.
func (m *EmbeddingModel) Embed(ctx context.Context, req *koine.EmbedRequest) (*koine.EmbedResponse, error) {
	params := openai.EmbeddingNewParams{
		Model: openai.EmbeddingModel(m.model),
		Input: openai.EmbeddingNewParamsInputUnion{OfArrayOfStrings: req.Inputs},
	}
	if req.Dimensions > 0 {
		params.Dimensions = openai.Int(int64(req.Dimensions))
	}
	if o, ok := req.ProviderOptions[Name].(EmbedOptions); ok && o.User != "" {
		params.User = openai.String(o.User)
	}
	resp, err := m.client.Embeddings.New(ctx, params)
	if err != nil {
		return nil, wrapErr(err)
	}
	out := &koine.EmbedResponse{
		Embeddings: make([]koine.Embedding, len(resp.Data)),
		Usage:      koine.Usage{InputTokens: int(resp.Usage.PromptTokens)},
		Model:      resp.Model,
		Provider:   Name,
	}
	for _, d := range resp.Data {
		vec := make(koine.Embedding, len(d.Embedding))
		for i, v := range d.Embedding {
			vec[i] = float32(v)
		}
		out.Embeddings[d.Index] = vec
	}
	return out, nil
}
