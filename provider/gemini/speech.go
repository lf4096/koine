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

var _ koine.SpeechModel = (*SpeechModel)(nil)

// SpeechModel builds the speech model.
func (p *Provider) SpeechModel(model string) *SpeechModel {
	return &SpeechModel{e: p.engine(model)}
}

func (m *SpeechModel) Model() string { return m.e.Model }

func (m *SpeechModel) Provider() string { return Name }

// GenerateSpeech performs one TTS call; output is raw PCM audio.
func (m *SpeechModel) GenerateSpeech(ctx context.Context, req *koine.SpeechRequest) (*koine.SpeechResponse, error) {
	return m.e.GenerateSpeech(ctx, req)
}
