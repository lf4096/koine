package openai

import (
	"bytes"
	"context"
	"encoding/json"

	"github.com/openai/openai-go/v3"

	"github.com/lf4096/koine"
)

// TranscriptionOptions carries OpenAI-specific transcription parameters. Pass
// via TranscriptionRequest.ProviderOptions[openai.Name].
type TranscriptionOptions struct {
	// ResponseFormat "verbose_json" populates Segments/Duration/Language;
	// whisper models only (gpt-4o-transcribe* reject it).
	ResponseFormat string
	// TimestampGranularities: "word" and/or "segment" (verbose_json only).
	TimestampGranularities []string
	Temperature            *float64
}

// TranscriptionModel transcribes audio via the OpenAI audio/transcriptions
// endpoint.
type TranscriptionModel struct {
	model  string
	client openai.Client
}

var _ koine.TranscriptionModel = (*TranscriptionModel)(nil)

// TranscriptionModel builds the transcription model.
func (p *Provider) TranscriptionModel(model string) *TranscriptionModel {
	return &TranscriptionModel{model: model, client: p.client}
}

func (m *TranscriptionModel) Model() string { return m.model }

func (m *TranscriptionModel) Provider() string { return Name }

// Transcribe performs one speech-to-text call.
func (m *TranscriptionModel) Transcribe(ctx context.Context, req *koine.TranscriptionRequest) (*koine.TranscriptionResponse, error) {
	params := openai.AudioTranscriptionNewParams{
		File:  openai.File(bytes.NewReader(req.Audio), "audio."+audioExtFromMIME(req.MIMEType), req.MIMEType),
		Model: openai.AudioModel(m.model),
	}
	if req.Language != "" {
		params.Language = openai.String(req.Language)
	}
	if req.Prompt != "" {
		params.Prompt = openai.String(req.Prompt)
	}
	if o, ok := req.ProviderOptions[Name].(TranscriptionOptions); ok {
		params.ResponseFormat = openai.AudioResponseFormat(o.ResponseFormat)
		params.TimestampGranularities = o.TimestampGranularities
		if o.Temperature != nil {
			params.Temperature = openai.Float(*o.Temperature)
		}
	}
	resp, err := m.client.Audio.Transcriptions.New(ctx, params)
	if err != nil {
		return nil, wrapErr(err)
	}
	out := &koine.TranscriptionResponse{
		Text:     resp.Text,
		Language: resp.Language,
		Duration: resp.Duration,
		Usage: koine.Usage{
			InputTokens:  int(resp.Usage.InputTokens),
			OutputTokens: int(resp.Usage.OutputTokens),
		},
		Raw:      json.RawMessage(resp.RawJSON()),
		Model:    m.model,
		Provider: Name,
	}
	if out.Language == "" && len(resp.Languages) > 0 {
		out.Language = resp.Languages[0].Code
	}
	for _, s := range resp.Segments {
		out.Segments = append(out.Segments, koine.Segment{Text: s.Text, Start: s.Start, End: s.End})
	}
	return out, nil
}

func audioExtFromMIME(mime string) string {
	switch mime {
	case "audio/mpeg", "audio/mp3":
		return "mp3"
	case "audio/mp4", "audio/m4a":
		return "m4a"
	case "audio/ogg":
		return "ogg"
	case "audio/flac":
		return "flac"
	case "audio/webm":
		return "webm"
	default:
		return "wav"
	}
}
