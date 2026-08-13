package koine

import "context"

// EmbeddingModel turns text into vectors for one model. Implementations
// only translate formats; transport stays in the provider SDK underneath.
type EmbeddingModel interface {
	Model() string
	Provider() string
	Embed(ctx context.Context, req *EmbedRequest) (*EmbedResponse, error)
}

// EmbedRequest is one embedding call over a batch of inputs.
type EmbedRequest struct {
	Inputs []string
	// Dimensions truncates the output vector where the model supports it.
	// 0 = model default.
	Dimensions int
	// Task hints the intended use; ignored where unsupported.
	Task            EmbedTask
	ProviderOptions map[string]any
}

// EmbedTask is a normalized hint for how a vector will be used. Providers
// that tune output per task map it; providers without the concept ignore it.
// Empty leaves the provider default.
type EmbedTask string

const (
	EmbedTaskQuery          EmbedTask = "query"
	EmbedTaskDocument       EmbedTask = "document"
	EmbedTaskSimilarity     EmbedTask = "similarity"
	EmbedTaskClassification EmbedTask = "classification"
	EmbedTaskClustering     EmbedTask = "clustering"
)

// EmbedResponse returns one vector per input, in input order. Usage reports
// input tokens only, and only where the provider reports them.
type EmbedResponse struct {
	Embeddings []Embedding `json:"embeddings"`
	Usage      Usage       `json:"usage"`
	// Model is the provider-reported model when returned, else the bound model.
	Model    string `json:"model,omitempty"`
	Provider string `json:"provider,omitempty"`
}

// Embedding is a dense vector. koine narrows to float32 at the edge: it is
// the native output type of the Gemini SDK and the storage type of mainstream
// vector stores, and extra float64 precision sits below model noise.
type Embedding []float32
