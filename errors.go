package koine

import (
	"errors"
	"fmt"
	"net/http"
	"strings"
)

// Error is the normalized failure surface across providers. The underlying
// provider SDK error stays reachable via Unwrap.
type Error struct {
	Provider string
	// Status is the HTTP status; 0 for transport-level failures.
	Status int
	// Code is the provider's error type or code when reported.
	Code    string
	Message string
	Err     error
}

func (e *Error) Error() string {
	if e.Status != 0 {
		return fmt.Sprintf("koine/%s: status %d: %s", e.Provider, e.Status, e.Message)
	}
	return fmt.Sprintf("koine/%s: %s", e.Provider, e.Message)
}

func (e *Error) Unwrap() error { return e.Err }

// Retryable reports whether the request may succeed on retry.
func (e *Error) Retryable() bool {
	switch e.Status {
	case http.StatusRequestTimeout, http.StatusTooManyRequests:
		return true
	}
	return e.Status >= 500
}

// Retryable reports whether err is a provider failure worth retrying
// (timeout, rate limit, or server error). It unwraps to *Error, so callers
// need no type assertion.
func Retryable(err error) bool {
	if kerr, ok := errors.AsType[*Error](err); ok {
		return kerr.Retryable()
	}
	return false
}

// ContextOverflow reports whether the provider rejected the request because
// the prompt does not fit the model's context window. It classifies by the
// provider code (OpenAI) or by the message text (Anthropic, Gemini, Vertex,
// and OpenAI-compatible gateways).
func (e *Error) ContextOverflow() bool {
	if e.Code == "context_length_exceeded" {
		return true
	}
	message := strings.ToLower(e.Message)
	for _, marker := range contextOverflowMarkers {
		if strings.Contains(message, marker) {
			return true
		}
	}
	return false
}

var contextOverflowMarkers = []string{
	"prompt is too long",
	"exceed context limit",
	"maximum context length",
	"exceeds the maximum number of input tokens",
	"but the model supports up to",
	"input is too long for requested model",
}

// ContextOverflow reports whether err is a context-window overflow. It
// unwraps to *Error, so callers need no type assertion.
func ContextOverflow(err error) bool {
	if kerr, ok := errors.AsType[*Error](err); ok {
		return kerr.ContextOverflow()
	}
	return false
}
