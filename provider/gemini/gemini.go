// Package gemini implements koine providers for the Gemini API, on top of the
// official google.golang.org/genai SDK. The translation layer is shared with
// provider/vertex, which speaks the same wire format through Vertex AI.
package gemini

import (
	"context"
	"net/http"

	"google.golang.org/genai"

	"github.com/lf4096/koine/internal/genaiengine"
)

// Name is the provider identifier, used in ProviderRaw and ProviderOptions.
const Name = "gemini"

// acceptRaw includes vertex: both providers serialize the identical
// genai.Part, so raw blocks replay across them without loss.
var acceptRaw = []string{Name, "vertex"}

// Provider is a configured Gemini API endpoint. Models derive from it and
// share its SDK client; build one Provider per credential.
type Provider struct {
	e genaiengine.Engine
}

// New builds a Provider. Without options, credentials come from the
// environment (GEMINI_API_KEY), as the official SDK defines.
func New(ctx context.Context, opts ...Option) (*Provider, error) {
	cfg := config{clientCfg: genai.ClientConfig{Backend: genai.BackendGeminiAPI}}
	for _, o := range opts {
		o(&cfg)
	}
	client := cfg.client
	if client == nil {
		var err error
		client, err = genai.NewClient(ctx, &cfg.clientCfg)
		if err != nil {
			return nil, genaiengine.ClientError(Name, err)
		}
	}
	return &Provider{e: genaiengine.Engine{Client: client, Name: Name, AcceptRaw: acceptRaw}}, nil
}

func (p *Provider) engine(model string) genaiengine.Engine {
	e := p.e
	e.Model = model
	return e
}

type config struct {
	clientCfg genai.ClientConfig
	client    *genai.Client
}

// Option configures a Provider.
type Option func(*config)

// WithAPIKey sets the API key.
func WithAPIKey(key string) Option {
	return func(c *config) { c.clientCfg.APIKey = key }
}

// WithBaseURL points the provider at a Gemini-compatible endpoint.
func WithBaseURL(url string) Option {
	return func(c *config) { c.clientCfg.HTTPOptions.BaseURL = url }
}

// WithRetry retries failed requests with backoff; the zero value takes the
// SDK defaults. Without it the SDK sends each request once.
func WithRetry(opts genai.HTTPRetryOptions) Option {
	return func(c *config) { c.clientCfg.HTTPOptions.RetryOptions = &opts }
}

// WithHTTPClient sets the underlying HTTP client.
func WithHTTPClient(hc *http.Client) Option {
	return func(c *config) { c.clientCfg.HTTPClient = hc }
}

// WithClient injects a preconfigured SDK client, overriding all other options.
func WithClient(client *genai.Client) Option {
	return func(c *config) { c.client = client }
}
