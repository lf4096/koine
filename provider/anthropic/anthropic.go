// Package anthropic implements the koine provider for the Anthropic Messages
// protocol, on top of the official anthropic-sdk-go.
package anthropic

import (
	"encoding/json"
	"errors"
	"net/http"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/option"

	"github.com/lf4096/koine"
)

// Name is the provider identifier, used in ProviderRaw and ProviderOptions.
const Name = "anthropic"

type config struct {
	reqOpts []option.RequestOption
	client  *anthropic.Client
}

// Option configures a model constructor.
type Option func(*config)

// WithAPIKey sets the API key.
func WithAPIKey(key string) Option {
	return func(c *config) { c.reqOpts = append(c.reqOpts, option.WithAPIKey(key)) }
}

// WithBaseURL points the provider at an Anthropic-compatible endpoint.
func WithBaseURL(url string) Option {
	return func(c *config) { c.reqOpts = append(c.reqOpts, option.WithBaseURL(url)) }
}

// WithHTTPClient sets the underlying HTTP client.
func WithHTTPClient(hc *http.Client) Option {
	return func(c *config) { c.reqOpts = append(c.reqOpts, option.WithHTTPClient(hc)) }
}

// WithClient injects a preconfigured SDK client, overriding all other options.
func WithClient(client anthropic.Client) Option {
	return func(c *config) { c.client = &client }
}

func wrapErr(err error) error {
	if apierr, ok := errors.AsType[*anthropic.Error](err); ok {
		return &koine.Error{
			Provider: Name,
			Status:   apierr.StatusCode,
			Code:     string(apierr.Type()),
			Message:  apiErrMessage(apierr),
			Err:      err,
		}
	}
	return &koine.Error{Provider: Name, Message: err.Error(), Err: err}
}

// apiErrMessage pulls the human message out of the Anthropic error envelope;
// the SDK surfaces it only inside the raw JSON body.
func apiErrMessage(apierr *anthropic.Error) string {
	var env struct {
		Error struct {
			Message string `json:"message"`
		} `json:"error"`
	}
	if json.Unmarshal([]byte(apierr.RawJSON()), &env) == nil && env.Error.Message != "" {
		return env.Error.Message
	}
	return apierr.Error()
}
