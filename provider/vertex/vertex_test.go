package vertex

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"testing"

	"cloud.google.com/go/auth"
	"google.golang.org/genai"

	"github.com/lf4096/koine"
)

func TestStreamThroughGateway(t *testing.T) {
	var paths []string
	var auths []string
	var body json.RawMessage
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		paths = append(paths, r.URL.Path)
		auths = append(auths, r.Header.Get("Authorization"))
		var buf json.RawMessage
		if err := json.NewDecoder(r.Body).Decode(&buf); err == nil {
			body = buf
		}
		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprint(w, "data: {\"candidates\":[{\"content\":{\"role\":\"model\",\"parts\":[{\"text\":\"Hi\"}]},\"finishReason\":\"STOP\",\"index\":0}],\"modelVersion\":\"gemini-test\",\"usageMetadata\":{\"promptTokenCount\":3,\"candidatesTokenCount\":1}}\r\n\r\n")
	}))
	defer server.Close()

	p, err := New(context.Background(),
		WithBaseURL(server.URL),
		WithAPIVersion("v1"),
		WithHeader("Authorization", "Bearer test-key"),
	)
	if err != nil {
		t.Fatal(err)
	}
	m := p.LanguageModel("gemini-test")
	// The raw block was minted by the gemini provider; vertex speaks the same
	// wire format and must replay it verbatim.
	req := &koine.LanguageRequest{
		Messages: []koine.Message{
			koine.UserText("hi"),
			{Role: koine.RoleAssistant, Blocks: koine.Blocks{
				&koine.ThinkingBlock{Text: "T", Signature: "c2ln", Raw: &koine.ProviderRaw{
					Provider: "gemini",
					JSON:     json.RawMessage(`{"text":"T","thought":true,"thoughtSignature":"c2ln"}`),
				}},
				&koine.TextBlock{Text: "prior answer"},
			}},
			koine.UserText("next"),
		},
	}
	stream, err := m.Stream(context.Background(), req)
	if err != nil {
		t.Fatal(err)
	}
	for stream.Next() {
	}
	if err := stream.Err(); err != nil {
		t.Fatal(err)
	}
	resp := stream.Response()
	if resp.Provider != Name || resp.Message.Text() != "Hi" {
		t.Errorf("resp = %+v", resp)
	}
	if len(paths) != 1 || !strings.HasPrefix(paths[0], "/v1/publishers/google/models/gemini-test") {
		t.Errorf("paths = %v", paths)
	}
	if auths[0] != "Bearer test-key" {
		t.Errorf("auth header = %q", auths[0])
	}
	var wire struct {
		Contents []struct {
			Parts []map[string]any `json:"parts"`
		} `json:"contents"`
	}
	if err := json.Unmarshal(body, &wire); err != nil {
		t.Fatal(err)
	}
	assistant := wire.Contents[1]
	if len(assistant.Parts) != 2 || assistant.Parts[0]["thoughtSignature"] != "c2ln" {
		t.Errorf("gemini raw block not replayed: %+v", assistant.Parts)
	}
}

func TestEmbedSplitsBatch(t *testing.T) {
	var bodies []string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var buf json.RawMessage
		if err := json.NewDecoder(r.Body).Decode(&buf); err == nil {
			bodies = append(bodies, string(buf))
		}
		w.Header().Set("Content-Type", "application/json")
		fmt.Fprint(w, `{"predictions":[{"embeddings":{"values":[0.5,0.6],"statistics":{"token_count":3}}}]}`)
	}))
	defer server.Close()

	p, err := New(context.Background(),
		WithBaseURL(server.URL),
		WithAPIVersion("v1"),
		WithHeader("Authorization", "Bearer test-key"),
	)
	if err != nil {
		t.Fatal(err)
	}
	m := p.EmbeddingModel("gemini-embedding-001")
	resp, err := m.Embed(context.Background(), &koine.EmbedRequest{
		Inputs: []string{"alpha", "beta"},
	})
	if err != nil {
		t.Fatal(err)
	}
	// The Vertex predict endpoint takes one content per call.
	if len(bodies) != 2 {
		t.Fatalf("expected 2 calls, got %d", len(bodies))
	}
	want := []koine.Embedding{{0.5, 0.6}, {0.5, 0.6}}
	if !reflect.DeepEqual(resp.Embeddings, want) {
		t.Errorf("embeddings = %v", resp.Embeddings)
	}
	if resp.Usage.InputTokens != 6 {
		t.Errorf("usage = %+v", resp.Usage)
	}
}

func TestAPIKeyWithProject(t *testing.T) {
	var path, key string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		path, key = r.URL.Path, r.Header.Get("x-goog-api-key")
		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprint(w, "data: {\"candidates\":[{\"content\":{\"role\":\"model\",\"parts\":[{\"text\":\"Hi\"}]},\"finishReason\":\"STOP\",\"index\":0}]}\r\n\r\n")
	}))
	defer server.Close()

	p, err := New(context.Background(), WithAPIKey("k"), WithProject("p"), WithLocation("us-central1"), WithBaseURL(server.URL), WithAPIVersion("v1"))
	if err != nil {
		t.Fatal(err)
	}
	if _, err := p.LanguageModel("gemini-test").Complete(context.Background(), &koine.LanguageRequest{Messages: []koine.Message{koine.UserText("hi")}}); err != nil {
		t.Fatal(err)
	}
	if path != "/v1/projects/p/locations/us-central1/publishers/google/models/gemini-test:streamGenerateContent" || key != "k" {
		t.Errorf("path = %q, x-goog-api-key = %q", path, key)
	}
}

func TestRetry(t *testing.T) {
	requests := 0
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		requests++
		if requests%2 == 1 {
			w.WriteHeader(http.StatusServiceUnavailable)
			fmt.Fprint(w, `{"error":{"code":503,"message":"busy","status":"UNAVAILABLE"}}`)
			return
		}
		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprint(w, "data: {\"candidates\":[{\"content\":{\"role\":\"model\",\"parts\":[{\"text\":\"Hi\"}]},\"finishReason\":\"STOP\",\"index\":0}]}\r\n\r\n")
	}))
	defer server.Close()
	zero := 0.0
	for _, retry := range []bool{false, true} {
		requests = 0
		opts := []Option{WithAPIVersion("v1"), WithBaseURL(server.URL)}
		if retry {
			opts = append(opts, WithRetry(genai.HTTPRetryOptions{Attempts: new(int32(2)), InitialDelay: &zero, Jitter: &zero}))
		}
		p, err := New(context.Background(), opts...)
		if err != nil {
			t.Fatal(err)
		}
		resp, err := p.LanguageModel("gemini-test").Complete(context.Background(), &koine.LanguageRequest{Messages: []koine.Message{koine.UserText("hi")}})
		if retry && (err != nil || resp.Message.Text() != "Hi" || requests != 2) {
			t.Errorf("with retry: err = %v, requests = %d", err, requests)
		}
		if !retry && (err == nil || requests != 1) {
			t.Errorf("without retry: err = %v, requests = %d", err, requests)
		}
	}
}

func TestClientErrorHidesSecrets(t *testing.T) {
	_, err := New(context.Background(),
		WithAPIKey("SECRET-KEY"),
		WithCredentials(auth.NewCredentials(&auth.CredentialsOptions{})),
		WithHeader("Authorization", "Bearer SECRET-TOKEN"),
	)
	if err == nil || !strings.Contains(err.Error(), "mutually exclusive") || strings.Contains(err.Error(), "SECRET") || errors.Unwrap(err) != nil {
		t.Errorf("err = %v", err)
	}
}
