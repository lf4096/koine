// Package vertex implements koine providers for Vertex AI, on top of the
// official google.golang.org/genai SDK. Vertex and the Gemini API share one
// wire format; this package differs from provider/gemini only in client
// configuration (project/location, express API key, or a self-authenticating
// gateway) and identity.
package vertex

import (
	"context"
	"net/http"

	"cloud.google.com/go/auth"
	"google.golang.org/genai"

	"github.com/lf4096/koine/internal/genaiengine"
)

// Name is the provider identifier, used in ProviderRaw and ProviderOptions.
const Name = "vertex"

// acceptRaw includes gemini: both providers serialize the identical
// genai.Part, so raw blocks replay across them without loss.
var acceptRaw = []string{Name, "gemini"}

// Provider is a configured Vertex AI endpoint. Models derive from it and
// share its SDK client; build one Provider per credential.
type Provider struct {
	e genaiengine.Engine
}

// New builds a Provider. Without options, project, location, and credentials
// come from the environment (GOOGLE_CLOUD_PROJECT, GOOGLE_CLOUD_LOCATION,
// Application Default Credentials), as the official SDK defines.
func New(ctx context.Context, opts ...Option) (*Provider, error) {
	cfg := config{clientCfg: genai.ClientConfig{Backend: genai.BackendVertexAI}}
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
	return &Provider{e: genaiengine.Engine{
		Client:    client,
		Name:      Name,
		AcceptRaw: acceptRaw,
		// The Vertex predict endpoint rejects multi-content embed calls.
		SplitEmbedBatch: true,
	}}, nil
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

// WithProject sets the GCP project id.
func WithProject(id string) Option {
	return func(c *config) { c.clientCfg.Project = id }
}

// WithLocation sets the GCP region (e.g. "us-central1", "global").
func WithLocation(location string) Option {
	return func(c *config) { c.clientCfg.Location = location }
}

// WithCredentials sets explicit Google credentials; without them the SDK uses
// Application Default Credentials.
func WithCredentials(creds *auth.Credentials) Option {
	return func(c *config) { c.clientCfg.Credentials = creds }
}

// WithAPIKey sends the key as x-goog-api-key instead of Google credentials.
// Alone it enables Vertex express mode; with WithProject and WithLocation the
// calls use project-scoped paths. Mutually exclusive with WithCredentials.
func WithAPIKey(key string) Option {
	return func(c *config) { c.clientCfg.APIKey = key }
}

// WithBaseURL points the provider at a Vertex-compatible endpoint, such as a
// gateway that proxies publishers/google/models/... paths. With no project,
// location, or API key set, the SDK skips Google credentials entirely and the
// gateway is expected to authenticate the request itself (see WithHeader).
func WithBaseURL(url string) Option {
	return func(c *config) { c.clientCfg.HTTPOptions.BaseURL = url }
}

// WithAPIVersion overrides the version path segment the SDK inserts after the
// base URL (Vertex default "v1beta1"). Gateways whose base path already pins a
// version need this to avoid a doubled segment.
func WithAPIVersion(version string) Option {
	return func(c *config) { c.clientCfg.HTTPOptions.APIVersion = version }
}

// WithHeader adds a static header to every request, e.g. an Authorization
// bearer token for gateways that do not take Google credentials.
func WithHeader(key, value string) Option {
	return func(c *config) {
		if c.clientCfg.HTTPOptions.Headers == nil {
			c.clientCfg.HTTPOptions.Headers = http.Header{}
		}
		c.clientCfg.HTTPOptions.Headers.Set(key, value)
	}
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
