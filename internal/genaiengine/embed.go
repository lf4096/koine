package genaiengine

import (
	"context"

	"google.golang.org/genai"

	"github.com/lf4096/koine"
)

// EmbedOptions carries embedding parameters specific to genai-based providers.
type EmbedOptions struct {
	// Title labels the input; only applies with EmbedTaskDocument.
	Title        string
	AutoTruncate bool
}

var taskTypes = map[koine.EmbedTask]string{
	koine.EmbedTaskQuery:          "RETRIEVAL_QUERY",
	koine.EmbedTaskDocument:       "RETRIEVAL_DOCUMENT",
	koine.EmbedTaskSimilarity:     "SEMANTIC_SIMILARITY",
	koine.EmbedTaskClassification: "CLASSIFICATION",
	koine.EmbedTaskClustering:     "CLUSTERING",
}

// Embed performs one embedContent call over the batch of inputs.
func (e *Engine) Embed(ctx context.Context, req *koine.EmbedRequest) (*koine.EmbedResponse, error) {
	config := &genai.EmbedContentConfig{TaskType: taskTypes[req.Task]}
	if req.Dimensions > 0 {
		config.OutputDimensionality = new(int32(req.Dimensions))
	}
	if o, ok := req.ProviderOptions[e.Name].(EmbedOptions); ok {
		config.Title = o.Title
		config.AutoTruncate = o.AutoTruncate
	}
	contents := make([]*genai.Content, len(req.Inputs))
	for i, input := range req.Inputs {
		contents[i] = genai.NewContentFromText(input, genai.RoleUser)
	}

	out := &koine.EmbedResponse{Model: e.Model, Provider: e.Name}
	batches := [][]*genai.Content{contents}
	if e.SplitEmbedBatch && len(contents) > 1 {
		batches = batches[:0]
		for _, c := range contents {
			batches = append(batches, []*genai.Content{c})
		}
	}
	var tokens float32
	for _, batch := range batches {
		resp, err := e.Client.Models.EmbedContent(ctx, e.Model, batch, config)
		if err != nil {
			return nil, e.wrapErr(err)
		}
		for _, emb := range resp.Embeddings {
			out.Embeddings = append(out.Embeddings, koine.Embedding(emb.Values))
			if emb.Statistics != nil {
				tokens += emb.Statistics.TokenCount
			}
		}
	}
	out.Usage.InputTokens = int(tokens)
	return out, nil
}
