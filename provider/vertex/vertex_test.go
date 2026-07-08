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

	m, err := NewLanguageModel(context.Background(),
		WithBaseURL(server.URL),
		WithAPIVersion("v1"),
		WithHeader("Authorization", "Bearer test-key"),
	)
	if err != nil {
		t.Fatal(err)
	}
	// The raw block was minted by the gemini provider; vertex speaks the same
	// wire format and must replay it verbatim.
	req := &koine.LanguageRequest{
		Model: "gemini-test",
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

	m, err := NewEmbeddingModel(context.Background(),
		WithBaseURL(server.URL),
		WithAPIVersion("v1"),
		WithHeader("Authorization", "Bearer test-key"),
	)
	if err != nil {
		t.Fatal(err)
	}
	resp, err := m.Embed(context.Background(), &koine.EmbedRequest{
		Model:  "gemini-embedding-001",
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

func TestExpressModeExclusiveWithProject(t *testing.T) {
	_, err := NewLanguageModel(context.Background(), WithAPIKey("k"), WithProject("p"))
	kerr, ok := errors.AsType[*koine.Error](err)
	if !ok || kerr.Provider != Name || !strings.Contains(kerr.Message, "mutually exclusive") {
		t.Errorf("err = %v", err)
	}
}
