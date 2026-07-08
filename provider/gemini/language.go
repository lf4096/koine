package gemini

import (
	"context"

	"github.com/lf4096/koine"
	"github.com/lf4096/koine/internal/genaiengine"
)

// LanguageOptions carries Gemini-specific request parameters. Pass via
// LanguageRequest.ProviderOptions[gemini.Name].
type LanguageOptions = genaiengine.LanguageOptions

// LanguageModel speaks the Gemini API chat surface.
type LanguageModel struct {
	e genaiengine.Engine
}

// NewLanguageModel builds the chat model. Without options, credentials come
// from the environment (GEMINI_API_KEY), as the official SDK defines.
func NewLanguageModel(ctx context.Context, opts ...Option) (*LanguageModel, error) {
	e, err := newEngine(ctx, opts)
	if err != nil {
		return nil, err
	}
	return &LanguageModel{e: e}, nil
}

func (m *LanguageModel) Name() string { return Name }

func (m *LanguageModel) Capabilities() koine.LanguageCapabilities {
	// Gemini caches implicitly; explicit cache objects are out of scope, so
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
