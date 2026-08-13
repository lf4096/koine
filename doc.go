// Package koine is a thin, fidelity-first Go library for LLM provider APIs --
// one tongue for every LLM.
//
// koine defines one canonical interface per model type -- LanguageModel,
// EmbeddingModel, ImageModel, SpeechModel, and TranscriptionModel -- and
// implements each only where a provider actually has the endpoint. Each
// provider package exposes New, which builds a Provider from credentials and
// endpoint options; models derive from it by name, so requests carry only
// per-call parameters.
// LanguageModel is the chat surface an agent loop consumes: one call with
// tools, thinking, prompt caching, and usage accounting, expressed in a
// canonical block-style message model. Everything above the model call --
// the loop, tool execution, retries, model catalogs -- stays with the caller.
//
// Coverage comes from protocol providers rather than per-vendor adapters:
// provider/anthropic (Claude), provider/openai (OpenAI and every
// OpenAI-compatible endpoint), provider/gemini, and provider/vertex (the same
// wire format as gemini, spoken through Vertex AI paths and auth). Each
// provider translates formats on top of the official vendor SDK and never
// touches HTTP itself.
//
// Fidelity beats normalization: blocks carry the provider's raw JSON
// (ProviderRaw) so thinking signatures and tool ids round-trip untouched when
// a conversation continues on the same provider, while cross-provider handoff
// converts lossily from the canonical fields.
package koine
