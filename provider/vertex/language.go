package vertex

import (
	"context"

	"github.com/lf4096/koine"
	"github.com/lf4096/koine/internal/genaiengine"
)

// LanguageOptions carries Vertex-specific request parameters. Pass via
// LanguageRequest.ProviderOptions[vertex.Name].
type LanguageOptions = genaiengine.LanguageOptions

// LanguageModel speaks the Vertex AI chat surface.
type LanguageModel struct {
	e genaiengine.Engine
}

var _ koine.LanguageModel = (*LanguageModel)(nil)

// LanguageModel builds the chat model.
func (p *Provider) LanguageModel(model string) *LanguageModel {
	return &LanguageModel{e: p.engine(model)}
}

func (m *LanguageModel) Model() string { return m.e.Model }

func (m *LanguageModel) Provider() string { return Name }

func (m *LanguageModel) Capabilities() koine.LanguageCapabilities {
	// Vertex caches implicitly; explicit cache objects are out of scope, so
	// CacheControl stays false.
	return koine.LanguageCapabilities{Thinking: true, ParallelToolCalls: true, Images: true, StructuredOutput: true}
}

// Complete performs one generateContent call and returns the final response.
func (m *LanguageModel) Complete(ctx context.Context, req *koine.LanguageRequest) (*koine.LanguageResponse, error) {
	s, err := m.Stream(ctx, req)
	if err != nil {
		return nil, err
	}
	return s.Collect()
}

// Stream performs one streaming generateContent call.
func (m *LanguageModel) Stream(ctx context.Context, req *koine.LanguageRequest) (*koine.LanguageStream, error) {
	return m.e.Stream(ctx, req)
}
