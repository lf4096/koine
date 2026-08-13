package openai

import (
	"context"
	"io"

	"github.com/openai/openai-go/v3"

	"github.com/lf4096/koine"
)

// SpeechModel synthesizes speech via the OpenAI audio/speech endpoint. (The
// SDK's openai.SpeechModel is an unrelated model-id string type.)
type SpeechModel struct {
	model  string
	client openai.Client
}

var _ koine.SpeechModel = (*SpeechModel)(nil)

// SpeechModel builds the speech model.
func (p *Provider) SpeechModel(model string) *SpeechModel {
	return &SpeechModel{model: model, client: p.client}
}

func (m *SpeechModel) Model() string { return m.model }

func (m *SpeechModel) Provider() string { return Name }

// GenerateSpeech performs one text-to-speech call. SpeechRequest.Language has
// no OpenAI parameter and is ignored.
func (m *SpeechModel) GenerateSpeech(ctx context.Context, req *koine.SpeechRequest) (*koine.SpeechResponse, error) {
	voice := req.Voice
	if voice == "" {
		voice = "alloy"
	}
	params := openai.AudioSpeechNewParams{
		Input:          req.Text,
		Model:          openai.SpeechModel(m.model),
		Voice:          openai.AudioSpeechNewParamsVoiceUnion{OfString: openai.String(voice)},
		ResponseFormat: openai.AudioSpeechNewParamsResponseFormat(req.Format),
	}
	if req.Instructions != "" {
		params.Instructions = openai.String(req.Instructions)
	}
	if req.Speed != 0 {
		params.Speed = openai.Float(req.Speed)
	}
	resp, err := m.client.Audio.Speech.New(ctx, params)
	if err != nil {
		return nil, wrapErr(err)
	}
	defer resp.Body.Close()
	audio, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, wrapErr(err)
	}
	return &koine.SpeechResponse{
		Audio:    audio,
		MIMEType: resp.Header.Get("Content-Type"),
		Model:    m.model,
		Provider: Name,
	}, nil
}
