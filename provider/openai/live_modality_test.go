package openai

import (
	"context"
	"encoding/json"
	"testing"

	"github.com/lf4096/koine"
	"github.com/lf4096/koine/internal/testenv"
)

func liveOpts(t *testing.T) []Option {
	t.Helper()
	key := testenv.Key(t, "KOINE_TEST_OPENAI_API_KEY")
	opts := []Option{WithAPIKey(key)}
	if base := testenv.Get("KOINE_TEST_OPENAI_BASE_URL", ""); base != "" {
		opts = append(opts, WithBaseURL(base))
	}
	return opts
}

func TestLiveEmbed(t *testing.T) {
	m := NewEmbeddingModel(liveOpts(t)...)
	resp, err := m.Embed(context.Background(), &koine.EmbedRequest{
		Model:  testenv.Get("KOINE_TEST_OPENAI_EMBED_MODEL", "text-embedding-3-small"),
		Inputs: []string{"the common tongue", "one tongue for every LLM"},
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(resp.Embeddings) != 2 || len(resp.Embeddings[0]) == 0 {
		t.Fatalf("embeddings shape: %d x %d", len(resp.Embeddings), len(resp.Embeddings[0]))
	}
	if resp.Usage.InputTokens == 0 {
		t.Errorf("usage not populated: %+v", resp.Usage)
	}
}

func TestLiveSpeechTranscribeRoundTrip(t *testing.T) {
	opts := liveOpts(t)
	ctx := context.Background()

	speech, err := NewSpeechModel(opts...).GenerateSpeech(ctx, &koine.SpeechRequest{
		Model:  testenv.Get("KOINE_TEST_OPENAI_TTS_MODEL", "gpt-4o-mini-tts"),
		Text:   "Koine speaks every tongue.",
		Voice:  "alloy",
		Format: "mp3",
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(speech.Audio) == 0 || speech.MIMEType == "" {
		t.Fatalf("speech = %d bytes, mime %q", len(speech.Audio), speech.MIMEType)
	}

	transcript, err := NewTranscriptionModel(opts...).Transcribe(ctx, &koine.TranscriptionRequest{
		Model:    testenv.Get("KOINE_TEST_OPENAI_STT_MODEL", "gpt-4o-mini-transcribe"),
		Audio:    speech.Audio,
		MIMEType: "audio/mpeg",
		Language: "en",
	})
	if err != nil {
		t.Fatal(err)
	}
	if transcript.Text == "" {
		t.Fatalf("empty transcript: %+v", transcript)
	}
	t.Logf("roundtrip transcript: %q", transcript.Text)
}

func TestLiveGenerateImage(t *testing.T) {
	m := NewImageModel(liveOpts(t)...)
	resp, err := m.GenerateImage(context.Background(), &koine.ImageRequest{
		Model:  testenv.Get("KOINE_TEST_OPENAI_IMAGE_MODEL", "gpt-image-1-mini"),
		Prompt: "a minimalist line drawing of an ancient greek scroll",
		Size:   "1024x1024",
		ProviderOptions: map[string]any{Name: ImageOptions{
			Quality: "low",
		}},
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(resp.Results) != 1 || resp.Results[0].Image == nil {
		t.Fatalf("results = %+v", resp.Results)
	}
	img := resp.Results[0].Image
	if len(img.Data) == 0 && img.URL == "" {
		t.Fatalf("image empty: %+v", img)
	}
}

func TestLiveStructuredOutput(t *testing.T) {
	m := NewLanguageModel(liveOpts(t)...)
	resp, err := m.Complete(context.Background(), &koine.LanguageRequest{
		Model:     testenv.Get("KOINE_TEST_OPENAI_MODEL", "gpt-4.1-mini"),
		MaxTokens: 128,
		Messages:  []koine.Message{koine.UserText("What is the capital of France?")},
		ResponseFormat: &koine.ResponseFormat{
			Schema: map[string]any{
				"type":                 "object",
				"properties":           map[string]any{"capital": map[string]any{"type": "string"}},
				"required":             []string{"capital"},
				"additionalProperties": false,
			},
			Strict: true,
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	var out struct {
		Capital string `json:"capital"`
	}
	if err := json.Unmarshal([]byte(resp.Message.Text()), &out); err != nil {
		t.Fatalf("output is not schema JSON: %v\n%s", err, resp.Message.Text())
	}
	if out.Capital == "" {
		t.Fatalf("empty capital: %s", resp.Message.Text())
	}
}
