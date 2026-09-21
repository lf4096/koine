package koine_test

import (
	"errors"
	"fmt"
	"testing"

	"github.com/lf4096/koine"
)

func TestErrorRetryable(t *testing.T) {
	cases := []struct {
		status int
		want   bool
	}{{429, true}, {408, true}, {500, true}, {503, true}, {400, false}, {401, false}, {0, false}}
	for _, c := range cases {
		e := &koine.Error{Provider: "x", Status: c.status, Message: "m"}
		if e.Retryable() != c.want {
			t.Errorf("status %d: Retryable() = %v, want %v", c.status, e.Retryable(), c.want)
		}
	}
}

func TestRetryable(t *testing.T) {
	wrapped := fmt.Errorf("call failed: %w", &koine.Error{Provider: "x", Status: 429, Message: "m"})
	if !koine.Retryable(wrapped) {
		t.Error("Retryable(wrapped 429) = false, want true")
	}
	if koine.Retryable(fmt.Errorf("call failed: %w", &koine.Error{Provider: "x", Status: 400, Message: "m"})) {
		t.Error("Retryable(400) = true, want false")
	}
	if koine.Retryable(errors.New("not a koine error")) {
		t.Error("Retryable(foreign error) = true, want false")
	}
}

func TestContextOverflow(t *testing.T) {
	cases := []struct {
		name string
		err  *koine.Error
		want bool
	}{
		{"anthropic prompt", &koine.Error{Provider: "anthropic", Status: 400, Code: "invalid_request_error", Message: "prompt is too long: 213467 tokens > 200000 maximum"}, true},
		{"anthropic max_tokens", &koine.Error{Provider: "anthropic", Status: 400, Code: "invalid_request_error", Message: "input length and `max_tokens` exceed context limit: 199999 + 21333 > 204698"}, true},
		{"openai code", &koine.Error{Provider: "openai", Status: 400, Code: "context_length_exceeded", Message: "This model's maximum context length is 128000 tokens."}, true},
		{"gemini", &koine.Error{Provider: "gemini", Status: 400, Code: "INVALID_ARGUMENT", Message: "The input token count (1200000) exceeds the maximum number of input tokens allowed (1048576)."}, true},
		{"vertex", &koine.Error{Provider: "vertex", Status: 400, Code: "INVALID_ARGUMENT", Message: "The input token count is 1200000 but the model supports up to 1048576."}, true},
		{"gateway code without status", &koine.Error{Provider: "openai", Status: 0, Code: "context_length_exceeded", Message: "context length exceeded"}, true},
		{"request too large", &koine.Error{Provider: "anthropic", Status: 413, Code: "request_too_large", Message: "Request exceeds the maximum allowed number of bytes"}, false},
		{"other 400", &koine.Error{Provider: "anthropic", Status: 400, Code: "invalid_request_error", Message: "messages: roles must alternate"}, false},
		{"rate limit", &koine.Error{Provider: "openai", Status: 429, Message: "slow down"}, false},
	}
	for _, c := range cases {
		if got := c.err.ContextOverflow(); got != c.want {
			t.Errorf("%s: ContextOverflow() = %v, want %v", c.name, got, c.want)
		}
	}
	wrapped := fmt.Errorf("call failed: %w", &koine.Error{Provider: "anthropic", Status: 400, Code: "invalid_request_error", Message: "prompt is too long: 213467 tokens > 200000 maximum"})
	if !koine.ContextOverflow(wrapped) {
		t.Error("ContextOverflow(wrapped) = false, want true")
	}
	if koine.ContextOverflow(errors.New("not a koine error")) {
		t.Error("ContextOverflow(foreign error) = true, want false")
	}
}
