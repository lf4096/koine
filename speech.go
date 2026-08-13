package koine

import (
	"context"
	"encoding/json"
)

// SpeechModel synthesizes speech from text.
type SpeechModel interface {
	Model() string
	Provider() string
	GenerateSpeech(ctx context.Context, req *SpeechRequest) (*SpeechResponse, error)
}

// SpeechRequest is one text-to-speech call.
type SpeechRequest struct {
	Text string
	// Voice is a provider voice id. Empty uses the provider default.
	Voice string
	// Format is the provider's output-format token (e.g. "mp3", "wav").
	// Providers with fixed output ignore it.
	Format string
	// Instructions steers delivery on models that accept it.
	Instructions string
	// Language is an ISO-639-1 code, where the provider takes one.
	Language string
	// Speed scales delivery rate where supported. 0 uses the default.
	Speed           float64
	ProviderOptions map[string]any
}

// SpeechResponse is synthesized audio. MIMEType is the response media type;
// Usage is populated where the provider reports it.
type SpeechResponse struct {
	Audio    []byte `json:"audio"`
	MIMEType string `json:"mime_type"`
	Usage    Usage  `json:"usage"`
	// Model is the provider-reported model when returned, else the bound model.
	Model    string `json:"model,omitempty"`
	Provider string `json:"provider,omitempty"`
}

// TranscriptionModel transcribes speech to text.
type TranscriptionModel interface {
	Model() string
	Provider() string
	Transcribe(ctx context.Context, req *TranscriptionRequest) (*TranscriptionResponse, error)
}

// TranscriptionRequest is one speech-to-text call. MIMEType identifies the
// audio container (audio/mpeg, audio/wav, ...).
type TranscriptionRequest struct {
	Audio    []byte
	MIMEType string
	// Language is an ISO-639-1 hint that improves accuracy.
	Language string
	// Prompt biases style or continues a prior segment.
	Prompt          string
	ProviderOptions map[string]any
}

// TranscriptionResponse is a transcript. Segments, Duration, and Usage are
// populated where the provider reports them; Raw carries the full provider
// response for fields koine does not normalize (word timestamps, logprobs).
type TranscriptionResponse struct {
	Text     string `json:"text"`
	Language string `json:"language,omitempty"`
	// Duration is the audio length in seconds; 0 when not reported.
	Duration float64         `json:"duration,omitempty"`
	Segments []Segment       `json:"segments,omitempty"`
	Usage    Usage           `json:"usage"`
	Raw      json.RawMessage `json:"raw,omitempty"`
	// Model is the bound model that produced the transcript.
	Model    string `json:"model,omitempty"`
	Provider string `json:"provider,omitempty"`
}

// Segment is a timestamped span of transcript; Start and End are seconds.
type Segment struct {
	Text  string  `json:"text"`
	Start float64 `json:"start"`
	End   float64 `json:"end"`
}
