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
