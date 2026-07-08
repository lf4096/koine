package gemini

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"testing"

	"github.com/lf4096/koine"
)

func jsonServer(t *testing.T, capture *json.RawMessage, path *string, body string) *httptest.Server {
	t.Helper()
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		*path = r.URL.Path
		var buf json.RawMessage
		if err := json.NewDecoder(r.Body).Decode(&buf); err == nil {
			*capture = buf
		}
		w.Header().Set("Content-Type", "application/json")
		fmt.Fprint(w, body)
	}))
	t.Cleanup(server.Close)
	return server
}

func TestEmbed(t *testing.T) {
	var body json.RawMessage
	var path string
	server := jsonServer(t, &body, &path, `{
		"embeddings":[{"values":[0.1,0.2]},{"values":[0.3,0.4]}]
	}`)
	m, err := NewEmbeddingModel(context.Background(), WithAPIKey("test"), WithBaseURL(server.URL))
	if err != nil {
		t.Fatal(err)
	}

	resp, err := m.Embed(context.Background(), &koine.EmbedRequest{
		Model:      "gemini-embedding-001",
		Inputs:     []string{"alpha", "beta"},
		Dimensions: 2,
		Task:       koine.EmbedTaskDocument,
		ProviderOptions: map[string]any{Name: EmbedOptions{
			Title: "doc title",
		}},
	})
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(path, "batchEmbedContents") {
		t.Errorf("path = %q", path)
	}
	wireStr := string(body)
	for _, want := range []string{`"RETRIEVAL_DOCUMENT"`, `"outputDimensionality":2`, `"doc title"`, `"alpha"`, `"beta"`} {
		if !strings.Contains(wireStr, want) {
			t.Errorf("wire missing %s:\n%s", want, wireStr)
		}
	}
	want := []koine.Embedding{{0.1, 0.2}, {0.3, 0.4}}
	if !reflect.DeepEqual(resp.Embeddings, want) {
		t.Errorf("embeddings = %v", resp.Embeddings)
	}
	if resp.Provider != Name {
		t.Errorf("provider = %q", resp.Provider)
	}
}

func TestGenerateImage(t *testing.T) {
	var body json.RawMessage
	var path string
	png := []byte{0x89, 'P', 'N', 'G'}
	server := jsonServer(t, &body, &path, `{
		"predictions":[
			{"bytesBase64Encoded":"`+base64.StdEncoding.EncodeToString(png)+`","mimeType":"image/png"},
			{"raiFilteredReason":"safety"}
		]
	}`)
	m, err := NewImageModel(context.Background(), WithAPIKey("test"), WithBaseURL(server.URL))
	if err != nil {
		t.Fatal(err)
	}

	resp, err := m.GenerateImage(context.Background(), &koine.ImageRequest{
		Model:       "imagen-4.0-generate-001",
		Prompt:      "a cat",
		N:           2,
		AspectRatio: "16:9",
		ProviderOptions: map[string]any{Name: ImageOptions{
			IncludeRAIReason: true,
		}},
	})
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(path, ":predict") {
		t.Errorf("path = %q", path)
	}
	wireStr := string(body)
	for _, want := range []string{`"a cat"`, `"sampleCount":2`, `"aspectRatio":"16:9"`, `"includeRaiReason":true`} {
		if !strings.Contains(wireStr, want) {
			t.Errorf("wire missing %s:\n%s", want, wireStr)
		}
	}
	if len(resp.Results) != 2 {
		t.Fatalf("results = %+v", resp.Results)
	}
	if img := resp.Results[0].Image; img == nil || !reflect.DeepEqual(img.Data, png) || img.MIMEType != "image/png" {
		t.Errorf("image = %+v", resp.Results[0].Image)
	}
	if r := resp.Results[1]; !r.Filtered || r.FilterReason != "safety" || r.Image != nil {
		t.Errorf("filtered result = %+v", r)
	}
}

func TestGenerateSpeech(t *testing.T) {
	var body json.RawMessage
	var path string
	pcm := []byte{1, 2, 3, 4}
	server := jsonServer(t, &body, &path, `{
		"candidates":[{"content":{"role":"model","parts":[{"inlineData":{"mimeType":"audio/L16;codec=pcm;rate=24000","data":"`+base64.StdEncoding.EncodeToString(pcm)+`"}}]},"finishReason":"STOP","index":0}],
		"modelVersion":"gemini-2.5-flash-preview-tts"
	}`)
	m, err := NewSpeechModel(context.Background(), WithAPIKey("test"), WithBaseURL(server.URL))
	if err != nil {
		t.Fatal(err)
	}

	resp, err := m.GenerateSpeech(context.Background(), &koine.SpeechRequest{
		Model: "gemini-2.5-flash-preview-tts",
		Text:  "hello",
		Voice: "Kore",
	})
	if err != nil {
		t.Fatal(err)
	}
	wireStr := string(body)
	for _, want := range []string{`"AUDIO"`, `"voiceName":"Kore"`, `"hello"`} {
		if !strings.Contains(wireStr, want) {
			t.Errorf("wire missing %s:\n%s", want, wireStr)
		}
	}
	if !reflect.DeepEqual(resp.Audio, pcm) || !strings.HasPrefix(resp.MIMEType, "audio/L16") {
		t.Errorf("resp = %+v", resp)
	}
}

func TestResponseFormatWire(t *testing.T) {
	var body json.RawMessage
	var path string
	server := jsonServer(t, &body, &path, `{"candidates":[{"content":{"role":"model","parts":[{"text":"{}"}]},"finishReason":"STOP","index":0}]}`)
	m, err := NewLanguageModel(context.Background(), WithAPIKey("test"), WithBaseURL(server.URL))
	if err != nil {
		t.Fatal(err)
	}

	stream, err := m.Stream(context.Background(), &koine.LanguageRequest{
		Model:    "gemini-test",
		Messages: []koine.Message{koine.UserText("hi")},
		ResponseFormat: &koine.ResponseFormat{
			Schema: map[string]any{"type": "object", "properties": map[string]any{"answer": map[string]any{"type": "string"}}},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	for stream.Next() {
	}
	stream.Close()

	var wire struct {
		GenerationConfig struct {
			ResponseMIMEType   string         `json:"responseMimeType"`
			ResponseJsonSchema map[string]any `json:"responseJsonSchema"`
		} `json:"generationConfig"`
	}
	if err := json.Unmarshal(body, &wire); err != nil {
		t.Fatal(err)
	}
	gc := wire.GenerationConfig
	if gc.ResponseMIMEType != "application/json" || gc.ResponseJsonSchema["type"] != "object" {
		t.Errorf("generationConfig = %+v", gc)
	}
}

func TestResponseFormatWithToolsErrors(t *testing.T) {
	m, err := NewLanguageModel(context.Background(), WithAPIKey("test"))
	if err != nil {
		t.Fatal(err)
	}
	_, err = m.Stream(context.Background(), &koine.LanguageRequest{
		Model:          "gemini-test",
		Messages:       []koine.Message{koine.UserText("hi")},
		Tools:          []koine.Tool{{Name: "f"}},
		ResponseFormat: &koine.ResponseFormat{Schema: map[string]any{"type": "object"}},
	})
	if err == nil || !strings.Contains(err.Error(), "cannot be combined with Tools") {
		t.Errorf("err = %v", err)
	}
}

func TestToolResultJSONNative(t *testing.T) {
	var body json.RawMessage
	var path string
	server := jsonServer(t, &body, &path, `{"candidates":[{"content":{"role":"model","parts":[{"text":"ok"}]},"finishReason":"STOP","index":0}]}`)
	m, err := NewLanguageModel(context.Background(), WithAPIKey("test"), WithBaseURL(server.URL))
	if err != nil {
		t.Fatal(err)
	}

	stream, err := m.Stream(context.Background(), &koine.LanguageRequest{
		Model: "gemini-test",
		Messages: []koine.Message{
			koine.UserText("hi"),
			{Role: koine.RoleAssistant, Blocks: koine.Blocks{
				&koine.ToolUseBlock{ID: "koine:0", Name: "get_weather", Input: json.RawMessage(`{}`)},
			}},
			koine.ToolResultJSON("koine:0", json.RawMessage(`{"temp":22,"unit":"C"}`), false),
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	for stream.Next() {
	}
	stream.Close()

	var wire struct {
		Contents []struct {
			Parts []struct {
				FunctionResponse *struct {
					Response map[string]any `json:"response"`
				} `json:"functionResponse"`
			} `json:"parts"`
		} `json:"contents"`
	}
	if err := json.Unmarshal(body, &wire); err != nil {
		t.Fatal(err)
	}
	fr := wire.Contents[2].Parts[0].FunctionResponse
	// The structured result must arrive as a native JSON object, not an
	// escaped string under "output".
	if fr == nil || fr.Response["temp"] != float64(22) || fr.Response["unit"] != "C" {
		t.Errorf("functionResponse = %+v", fr)
	}
}
