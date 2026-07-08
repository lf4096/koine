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
	client openai.Client
}

// NewEmbeddingModel builds the embedding model; options as NewLanguageModel.
func NewEmbeddingModel(opts ...Option) *EmbeddingModel {
	return &EmbeddingModel{client: newClient(opts)}
}

func (m *EmbeddingModel) Name() string { return Name }

// Embed performs one embeddings call. EmbedRequest.Task has no OpenAI
// equivalent and is ignored.
func (m *EmbeddingModel) Embed(ctx context.Context, req *koine.EmbedRequest) (*koine.EmbedResponse, error) {
	params := openai.EmbeddingNewParams{
		Model: openai.EmbeddingModel(req.Model),
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
