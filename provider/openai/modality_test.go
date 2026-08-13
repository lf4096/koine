package openai

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"reflect"
	"testing"

	"github.com/lf4096/koine"
)

func jsonServer(t *testing.T, capture *json.RawMessage, contentType, body string) *httptest.Server {
	t.Helper()
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var buf json.RawMessage
		if err := json.NewDecoder(r.Body).Decode(&buf); err == nil {
			*capture = buf
		}
		w.Header().Set("Content-Type", contentType)
		fmt.Fprint(w, body)
	}))
	t.Cleanup(server.Close)
	return server
}

func TestEmbed(t *testing.T) {
	var body json.RawMessage
	// Indices arrive shuffled; vectors must land in input order.
	server := jsonServer(t, &body, "application/json", `{
		"object":"list",
		"data":[
			{"object":"embedding","index":1,"embedding":[0.3,0.4]},
			{"object":"embedding","index":0,"embedding":[0.1,0.2]}
		],
		"model":"text-embedding-3-small",
		"usage":{"prompt_tokens":7,"total_tokens":7}
	}`)
	m := New(WithAPIKey("test"), WithBaseURL(server.URL)).EmbeddingModel("text-embedding-3-small")

	resp, err := m.Embed(context.Background(), &koine.EmbedRequest{
		Inputs:     []string{"alpha", "beta"},
		Dimensions: 2,
		Task:       koine.EmbedTaskQuery,
	})
	if err != nil {
		t.Fatal(err)
	}
	var wire struct {
		Model      string   `json:"model"`
		Input      []string `json:"input"`
		Dimensions int      `json:"dimensions"`
	}
	if err := json.Unmarshal(body, &wire); err != nil {
		t.Fatal(err)
	}
	if wire.Model != "text-embedding-3-small" || !reflect.DeepEqual(wire.Input, []string{"alpha", "beta"}) || wire.Dimensions != 2 {
		t.Errorf("wire = %+v", wire)
	}
	want := []koine.Embedding{{0.1, 0.2}, {0.3, 0.4}}
	if !reflect.DeepEqual(resp.Embeddings, want) {
		t.Errorf("embeddings = %v, want %v", resp.Embeddings, want)
	}
	if resp.Usage.InputTokens != 7 || resp.Model != "text-embedding-3-small" || resp.Provider != Name {
		t.Errorf("meta = %+v", resp)
	}
}

func TestGenerateImage(t *testing.T) {
	var body json.RawMessage
	png := []byte{0x89, 'P', 'N', 'G'}
	server := jsonServer(t, &body, "application/json", `{
		"created":1,
		"data":[{"b64_json":"`+base64.StdEncoding.EncodeToString(png)+`","revised_prompt":"a nicer cat"}],
		"output_format":"png",
		"usage":{"input_tokens":5,"output_tokens":100,"total_tokens":105}
	}`)
	m := New(WithAPIKey("test"), WithBaseURL(server.URL)).ImageModel("gpt-image-2")

	resp, err := m.GenerateImage(context.Background(), &koine.ImageRequest{
		Prompt: "a cat",
		Size:   "1024x1024",
		ProviderOptions: map[string]any{Name: ImageOptions{
			Quality: "high",
		}},
	})
	if err != nil {
		t.Fatal(err)
	}
	var wire struct {
		Prompt  string `json:"prompt"`
		Model   string `json:"model"`
		N       int    `json:"n"`
		Size    string `json:"size"`
		Quality string `json:"quality"`
	}
	if err := json.Unmarshal(body, &wire); err != nil {
		t.Fatal(err)
	}
	if wire.Prompt != "a cat" || wire.Model != "gpt-image-2" || wire.N != 1 || wire.Size != "1024x1024" || wire.Quality != "high" {
		t.Errorf("wire = %+v", wire)
	}
	if len(resp.Results) != 1 {
		t.Fatalf("results = %+v", resp.Results)
	}
	r := resp.Results[0]
	if !reflect.DeepEqual(r.Image.Data, png) || r.Image.MIMEType != "image/png" || r.RevisedPrompt != "a nicer cat" {
		t.Errorf("result = %+v image = %+v", r, r.Image)
	}
	if resp.Usage.InputTokens != 5 || resp.Usage.OutputTokens != 100 || resp.Model != "gpt-image-2" {
		t.Errorf("usage = %+v model = %q", resp.Usage, resp.Model)
	}
}

func TestEditImageMultipart(t *testing.T) {
	var gotContentType string
	var fields map[string][]string
	var fileParts []string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotContentType = r.Header.Get("Content-Type")
		if err := r.ParseMultipartForm(32 << 20); err != nil {
			t.Errorf("parse multipart: %v", err)
		}
		fields = r.MultipartForm.Value
		for key, headers := range r.MultipartForm.File {
			for _, h := range headers {
				fileParts = append(fileParts, key+":"+h.Filename)
			}
		}
		w.Header().Set("Content-Type", "application/json")
		fmt.Fprint(w, `{"created":1,"data":[{"url":"https://example.com/out.png"}]}`)
	}))
	defer server.Close()
	m := New(WithAPIKey("test"), WithBaseURL(server.URL)).ImageModel("gpt-image-2")

	resp, err := m.GenerateImage(context.Background(), &koine.ImageRequest{
		Prompt: "add a hat",
		Images: []*koine.ImageBlock{{MIMEType: "image/png", Data: []byte{1, 2}}},
		Mask:   &koine.ImageBlock{MIMEType: "image/png", Data: []byte{3}},
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(gotContentType) < 19 || gotContentType[:19] != "multipart/form-data" {
		t.Errorf("content type = %q", gotContentType)
	}
	if got := fields["prompt"]; len(got) != 1 || got[0] != "add a hat" {
		t.Errorf("prompt field = %v", fields)
	}
	if got := fields["model"]; len(got) != 1 || got[0] != "gpt-image-2" {
		t.Errorf("model field = %v", fields)
	}
	if len(fileParts) != 2 {
		t.Errorf("file parts = %v", fileParts)
	}
	if resp.Results[0].Image.URL != "https://example.com/out.png" {
		t.Errorf("result = %+v", resp.Results[0].Image)
	}
}

func TestGenerateSpeech(t *testing.T) {
	var body json.RawMessage
	audio := []byte("MP3DATA")
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var buf json.RawMessage
		if err := json.NewDecoder(r.Body).Decode(&buf); err == nil {
			body = buf
		}
		w.Header().Set("Content-Type", "audio/mpeg")
		w.Write(audio)
	}))
	defer server.Close()
	m := New(WithAPIKey("test"), WithBaseURL(server.URL)).SpeechModel("gpt-4o-mini-tts")

	resp, err := m.GenerateSpeech(context.Background(), &koine.SpeechRequest{
		Text:         "hello",
		Voice:        "coral",
		Format:       "mp3",
		Instructions: "cheerful",
		Speed:        1.5,
	})
	if err != nil {
		t.Fatal(err)
	}
	var wire struct {
		Input          string  `json:"input"`
		Model          string  `json:"model"`
		Voice          string  `json:"voice"`
		ResponseFormat string  `json:"response_format"`
		Instructions   string  `json:"instructions"`
		Speed          float64 `json:"speed"`
	}
	if err := json.Unmarshal(body, &wire); err != nil {
		t.Fatal(err)
	}
	if wire.Input != "hello" || wire.Model != "gpt-4o-mini-tts" || wire.Voice != "coral" || wire.ResponseFormat != "mp3" || wire.Instructions != "cheerful" || wire.Speed != 1.5 {
		t.Errorf("wire = %+v", wire)
	}
	if string(resp.Audio) != "MP3DATA" || resp.MIMEType != "audio/mpeg" || resp.Model != "gpt-4o-mini-tts" || resp.Provider != Name {
		t.Errorf("resp = %+v", resp)
	}
}

func TestTranscribe(t *testing.T) {
	var fields map[string][]string
	var fileNames []string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if err := r.ParseMultipartForm(32 << 20); err != nil {
			t.Errorf("parse multipart: %v", err)
		}
		fields = r.MultipartForm.Value
		for _, headers := range r.MultipartForm.File {
			for _, h := range headers {
				fileNames = append(fileNames, h.Filename)
			}
		}
		w.Header().Set("Content-Type", "application/json")
		fmt.Fprint(w, `{
			"text":"hello world",
			"language":"english",
			"duration":2.5,
			"segments":[{"id":0,"seek":0,"start":0,"end":2.5,"text":"hello world","tokens":[],"temperature":0,"avg_logprob":-0.2,"compression_ratio":1.0,"no_speech_prob":0.01}]
		}`)
	}))
	defer server.Close()
	m := New(WithAPIKey("test"), WithBaseURL(server.URL)).TranscriptionModel("whisper-1")

	resp, err := m.Transcribe(context.Background(), &koine.TranscriptionRequest{
		Audio:    []byte("RIFFdata"),
		MIMEType: "audio/wav",
		Language: "en",
		ProviderOptions: map[string]any{Name: TranscriptionOptions{
			ResponseFormat: "verbose_json",
		}},
	})
	if err != nil {
		t.Fatal(err)
	}
	if got := fields["model"]; len(got) != 1 || got[0] != "whisper-1" {
		t.Errorf("model field = %v", fields)
	}
	if got := fields["language"]; len(got) != 1 || got[0] != "en" {
		t.Errorf("language field = %v", fields)
	}
	if got := fields["response_format"]; len(got) != 1 || got[0] != "verbose_json" {
		t.Errorf("response_format field = %v", fields)
	}
	if len(fileNames) != 1 || fileNames[0] != "audio.wav" {
		t.Errorf("file names = %v", fileNames)
	}
	if resp.Text != "hello world" || resp.Language != "english" || resp.Duration != 2.5 {
		t.Errorf("resp = %+v", resp)
	}
	want := []koine.Segment{{Text: "hello world", Start: 0, End: 2.5}}
	if !reflect.DeepEqual(resp.Segments, want) {
		t.Errorf("segments = %+v", resp.Segments)
	}
	if len(resp.Raw) == 0 {
		t.Error("raw payload missing")
	}
}

func TestResponseFormatWire(t *testing.T) {
	var body json.RawMessage
	server := sseServer(t, &body, streamFixture...)
	d := New(WithAPIKey("test"), WithBaseURL(server.URL)).LanguageModel("gpt-test")

	stream, err := d.Stream(context.Background(), &koine.LanguageRequest{
		Messages: []koine.Message{koine.UserText("hi")},
		ResponseFormat: &koine.ResponseFormat{
			Schema:      map[string]any{"type": "object", "properties": map[string]any{"answer": map[string]any{"type": "string"}}},
			Description: "the answer",
			Strict:      true,
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	for stream.Next() {
	}
	stream.Close()

	var wire struct {
		ResponseFormat struct {
			Type       string `json:"type"`
			JSONSchema struct {
				Name        string         `json:"name"`
				Description string         `json:"description"`
				Strict      bool           `json:"strict"`
				Schema      map[string]any `json:"schema"`
			} `json:"json_schema"`
		} `json:"response_format"`
	}
	if err := json.Unmarshal(body, &wire); err != nil {
		t.Fatal(err)
	}
	rf := wire.ResponseFormat
	if rf.Type != "json_schema" || rf.JSONSchema.Name != "response" || !rf.JSONSchema.Strict || rf.JSONSchema.Description != "the answer" || rf.JSONSchema.Schema["type"] != "object" {
		t.Errorf("response_format = %+v", rf)
	}
}

func TestResponseFormatSchemaless(t *testing.T) {
	var body json.RawMessage
	server := sseServer(t, &body, streamFixture...)
	d := New(WithAPIKey("test"), WithBaseURL(server.URL)).LanguageModel("gpt-test")

	stream, err := d.Stream(context.Background(), &koine.LanguageRequest{
		Messages:       []koine.Message{koine.UserText("hi")},
		ResponseFormat: &koine.ResponseFormat{},
	})
	if err != nil {
		t.Fatal(err)
	}
	for stream.Next() {
	}
	stream.Close()

	var wire struct {
		ResponseFormat struct {
			Type string `json:"type"`
		} `json:"response_format"`
	}
	if err := json.Unmarshal(body, &wire); err != nil {
		t.Fatal(err)
	}
	if wire.ResponseFormat.Type != "json_object" {
		t.Errorf("response_format = %+v", wire.ResponseFormat)
	}
}

func TestToolResultJSONWire(t *testing.T) {
	var body json.RawMessage
	server := sseServer(t, &body, streamFixture...)
	d := New(WithAPIKey("test"), WithBaseURL(server.URL)).LanguageModel("gpt-test")

	stream, err := d.Stream(context.Background(), &koine.LanguageRequest{
		Messages: []koine.Message{
			koine.UserText("hi"),
			{Role: koine.RoleAssistant, Blocks: koine.Blocks{
				&koine.ToolUseBlock{ID: "call_1", Name: "get_weather", Input: json.RawMessage(`{}`)},
			}},
			koine.ToolResultJSON("call_1", json.RawMessage(`{"temp":22}`), false),
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	for stream.Next() {
	}
	stream.Close()

	var wire struct {
		Messages []struct {
			Role    string `json:"role"`
			Content any    `json:"content"`
		} `json:"messages"`
	}
	if err := json.Unmarshal(body, &wire); err != nil {
		t.Fatal(err)
	}
	last := wire.Messages[len(wire.Messages)-1]
	if last.Role != "tool" || last.Content != `{"temp":22}` {
		t.Errorf("tool message = %+v", last)
	}
}
