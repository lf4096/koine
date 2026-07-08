package koine

import (
	"errors"
	"fmt"
	"net/http"
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
