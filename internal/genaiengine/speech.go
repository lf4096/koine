package genaiengine

import (
	"context"

	"google.golang.org/genai"

	"github.com/lf4096/koine"
)

// GenerateSpeech performs one generateContent call with audio output; genai
// has no dedicated TTS endpoint. Output is raw PCM (audio/L16); Format and
// Speed have no genai knob and are ignored.
func (e *Engine) GenerateSpeech(ctx context.Context, req *koine.SpeechRequest) (*koine.SpeechResponse, error) {
	config := &genai.GenerateContentConfig{
		ResponseModalities: []string{"AUDIO"},
		SpeechConfig:       &genai.SpeechConfig{LanguageCode: req.Language},
	}
	if req.Voice != "" {
		config.SpeechConfig.VoiceConfig = &genai.VoiceConfig{
			PrebuiltVoiceConfig: &genai.PrebuiltVoiceConfig{VoiceName: req.Voice},
		}
	}
	text := req.Text
	if req.Instructions != "" {
		// No instructions field exists; steer via the prompt itself.
		text = req.Instructions + "\n\n" + text
	}
	contents := []*genai.Content{genai.NewContentFromText(text, genai.RoleUser)}
	resp, err := e.Client.Models.GenerateContent(ctx, e.Model, contents, config)
	if err != nil {
		return nil, e.wrapErr(err)
	}
	var usage koine.Usage
	if u := resp.UsageMetadata; u != nil {
		usage = koine.Usage{
			InputTokens:  int(u.PromptTokenCount),
			OutputTokens: int(u.CandidatesTokenCount),
		}
	}
	model := resp.ModelVersion
	if model == "" {
		model = e.Model
	}
	for _, cand := range resp.Candidates {
		if cand.Content == nil {
			continue
		}
		for _, part := range cand.Content.Parts {
			if part.InlineData != nil {
				return &koine.SpeechResponse{
					Audio:    part.InlineData.Data,
					MIMEType: part.InlineData.MIMEType,
					Usage:    usage,
					Model:    model,
					Provider: e.Name,
				}, nil
			}
		}
	}
	return nil, &koine.Error{Provider: e.Name, Message: "no audio in response"}
}
