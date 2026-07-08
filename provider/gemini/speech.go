package gemini

import (
	"context"

	"github.com/lf4096/koine"
	"github.com/lf4096/koine/internal/genaiengine"
)

// SpeechModel synthesizes speech via Gemini TTS models.
type SpeechModel struct {
	e genaiengine.Engine
}

// NewSpeechModel builds the speech model; options as NewLanguageModel.
func NewSpeechModel(ctx context.Context, opts ...Option) (*SpeechModel, error) {
	e, err := newEngine(ctx, opts)
	if err != nil {
		return nil, err
	}
	return &SpeechModel{e: e}, nil
}

func (m *SpeechModel) Name() string { return Name }

// GenerateSpeech performs one TTS call; output is raw PCM audio.
func (m *SpeechModel) GenerateSpeech(ctx context.Context, req *koine.SpeechRequest) (*koine.SpeechResponse, error) {
	return m.e.GenerateSpeech(ctx, req)
}
