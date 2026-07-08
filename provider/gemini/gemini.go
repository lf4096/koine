// Package gemini implements koine providers for the Gemini API, on top of the
// official google.golang.org/genai SDK. The translation layer is shared with
// provider/vertex, which speaks the same wire format through Vertex AI.
package gemini

import (
	"context"
	"net/http"

	"google.golang.org/genai"

	"github.com/lf4096/koine"
	"github.com/lf4096/koine/internal/genaiengine"
)

// Name is the provider identifier, used in ProviderRaw and ProviderOptions.
const Name = "gemini"

// acceptRaw includes vertex: both providers serialize the identical
// genai.Part, so raw blocks replay across them without loss.
var acceptRaw = []string{Name, "vertex"}

type config struct {
	clientCfg genai.ClientConfig
	client    *genai.Client
}

// Option configures a model constructor.
type Option func(*config)

// WithAPIKey sets the API key.
func WithAPIKey(key string) Option {
	return func(c *config) { c.clientCfg.APIKey = key }
}

// WithBaseURL points the provider at a Gemini-compatible endpoint.
func WithBaseURL(url string) Option {
	return func(c *config) { c.clientCfg.HTTPOptions.BaseURL = url }
}

// WithHTTPClient sets the underlying HTTP client.
func WithHTTPClient(hc *http.Client) Option {
	return func(c *config) { c.clientCfg.HTTPClient = hc }
}

// WithClient injects a preconfigured SDK client, overriding all other options.
func WithClient(client *genai.Client) Option {
	return func(c *config) { c.client = client }
}

func newEngine(ctx context.Context, opts []Option) (genaiengine.Engine, error) {
	cfg := config{clientCfg: genai.ClientConfig{Backend: genai.BackendGeminiAPI}}
	for _, o := range opts {
		o(&cfg)
	}
	client := cfg.client
	if client == nil {
		var err error
		client, err = genai.NewClient(ctx, &cfg.clientCfg)
		if err != nil {
			return genaiengine.Engine{}, &koine.Error{Provider: Name, Message: err.Error(), Err: err}
		}
	}
	return genaiengine.Engine{Client: client, Name: Name, AcceptRaw: acceptRaw}, nil
}
