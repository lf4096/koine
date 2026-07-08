// Package genaiengine is the shared translation engine for providers built on
// google.golang.org/genai: the Gemini API and Vertex AI expose identical
// request/response types, so provider/gemini and provider/vertex both delegate
// here and differ only in client configuration and identity.
package genaiengine

import (
	"errors"
	"slices"

	"google.golang.org/genai"

	"github.com/lf4096/koine"
)

// Engine translates between the canonical koine model and the genai SDK for
// one provider identity.
type Engine struct {
	Client *genai.Client
	// Name is stamped on responses, errors, and ProviderRaw.
	Name string
	// AcceptRaw lists provider names whose ProviderRaw this engine replays
	// verbatim. gemini and vertex speak the same wire format, so each accepts
	// the other's raw blocks; anything else converts lossily.
	AcceptRaw []string
	// SplitEmbedBatch sends one embed call per input, for backends whose
	// predict endpoint rejects multi-content requests (Vertex).
	SplitEmbedBatch bool
}

func (e *Engine) acceptsRaw(raw *koine.ProviderRaw) bool {
	return raw != nil && slices.Contains(e.AcceptRaw, raw.Provider)
}

func (e *Engine) wrapErr(err error) error {
	if apierr, ok := errors.AsType[genai.APIError](err); ok {
		return &koine.Error{
			Provider: e.Name,
			Status:   apierr.Code,
			Code:     apierr.Status,
			Message:  apierr.Message,
			Err:      err,
		}
	}
	return &koine.Error{Provider: e.Name, Message: err.Error(), Err: err}
}
