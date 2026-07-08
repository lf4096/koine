// Package openai implements koine providers for the OpenAI API, on top of the
// official openai-go SDK: Chat Completions plus the embedding, image, speech,
// and transcription endpoints. With a custom base URL the chat surface covers
// every OpenAI-compatible endpoint (DeepSeek, Kimi, Qwen, GLM, ...),
// including their reasoning_content thinking extension.
package openai

import (
	"errors"
	"net/http"

	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/option"

	"github.com/lf4096/koine"
)

// Name is the provider identifier, used in ProviderRaw and ProviderOptions.
const Name = "openai"

type config struct {
	reqOpts []option.RequestOption
	client  *openai.Client
}

// Option configures a model constructor.
type Option func(*config)

// WithAPIKey sets the API key.
func WithAPIKey(key string) Option {
	return func(c *config) { c.reqOpts = append(c.reqOpts, option.WithAPIKey(key)) }
}

// WithBaseURL points the provider at an OpenAI-compatible endpoint.
func WithBaseURL(url string) Option {
	return func(c *config) { c.reqOpts = append(c.reqOpts, option.WithBaseURL(url)) }
}

// WithHTTPClient sets the underlying HTTP client.
func WithHTTPClient(hc *http.Client) Option {
	return func(c *config) { c.reqOpts = append(c.reqOpts, option.WithHTTPClient(hc)) }
}

// WithClient injects a preconfigured SDK client, overriding all other options.
func WithClient(client openai.Client) Option {
	return func(c *config) { c.client = &client }
}

func newClient(opts []Option) openai.Client {
	var cfg config
	for _, o := range opts {
		o(&cfg)
	}
	if cfg.client != nil {
		return *cfg.client
	}
	return openai.NewClient(cfg.reqOpts...)
}

func wrapErr(err error) error {
	if apierr, ok := errors.AsType[*openai.Error](err); ok {
		return &koine.Error{
			Provider: Name,
			Status:   apierr.StatusCode,
			Code:     apierr.Code,
			Message:  apierr.Message,
			Err:      err,
		}
	}
	return &koine.Error{Provider: Name, Message: err.Error(), Err: err}
}
