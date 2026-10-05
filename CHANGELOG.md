# Changelog

All notable changes to this project are documented here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- **Breaking**: anthropic maps `Thinking.Effort` to adaptive thinking with `output_config.effort` instead of a token budget, as newer Claude models reject budgets. Models before Claude Opus 4.6 and Sonnet 4.6 reject adaptive thinking; pass `BudgetTokens` for them.
- anthropic asks for summarized thinking text whenever thinking is requested, since newer Claude models omit it by default.

## [0.3.0] - 2026-09-30

### Added

- `LanguageCapabilities.StopSequences` reports whether a model accepts `StopSequences`; it is false only on openai `LanguageModel`.
- openai `LanguageOptions.NoReasoningReplay` stops `ChatCompletionsModel` from sending vendor reasoning back, for endpoints that reject message fields they do not know.
- anthropic `LanguageOptions.Thinking` replaces the normalized thinking mapping with a native thinking config, and `LanguageOptions.Effort` sets `output_config.effort`.
- gemini and vertex `WithRetry` turns on the SDK's retries, which are off by default.

### Changed

- **Breaking**: openai `Provider.LanguageModel(model)` speaks the Responses API; Chat Completions moves to `Provider.ChatCompletionsModel(model)`, for endpoints that do not serve Responses. `LanguageModel` rejects `StopSequences`, which the Responses API does not accept.
- **Breaking**: openai sends `ThinkingMax`, and budgets above 24576 tokens, as `max` instead of `xhigh`; models without `max` reject it.
- vertex `WithAPIKey` can be combined with `WithProject` and `WithLocation`.
- Dependencies: openai-go v3.68.0, anthropic-sdk-go v1.76.0, genai v1.71.0.

### Removed

- **Breaking**: gemini and vertex `ImageModel` and `ImageOptions` (Imagen), which the Gemini API no longer serves and the SDK deprecates. Gemini image models generate images through `LanguageModel`.

### Fixed

- openai `ChatCompletionsModel` sends vendor reasoning back on replay, which DeepSeek and Kimi require on tool-call turns, and also reads reasoning streamed in a `reasoning` field instead of `reasoning_content`.
- openai reads the code and message of an error that arrives inside the stream, so `koine.ContextOverflow` recognizes a context overflow reported mid-stream.
- openai streams no longer fail with `unexpected end of JSON input` when the endpoint sends SSE keepalive comments.
- openai reports cache writes in `Usage.CacheWriteTokens` and leaves them out of `Usage.InputTokens`.
- openai transcription fills `Language` on models that return detected languages as a list (gpt-transcribe).
- anthropic reports the `model_context_window_exceeded` stop reason as `StopMaxTokens` instead of `StopOther`.
- anthropic fills `Usage.ReasoningTokens` on the final response, not only on the streamed usage event.
- gemini and vertex return an error when a stream breaks off mid-response, instead of a truncated response.
- gemini and vertex keep the thought signatures of generated images in the new `ImageBlock.Raw`; Gemini 3 image models reject a follow-up turn without them.
- gemini and vertex construction errors no longer include the API key or header tokens.

## [0.2.1] - 2026-09-21

### Added

- `koine.ContextOverflow(err)` and `(*Error).ContextOverflow()` classify a context-window overflow across providers.

### Fixed

- openai and gemini/vertex reported the whole prompt in `Usage.InputTokens` while also reporting the cached part in `CacheReadTokens`, so cached tokens were counted twice; `InputTokens` now excludes cache hits on every provider.

## [0.2.0] - 2026-08-13

### Added

- `ThinkingNone` disables reasoning explicitly (`reasoning_effort: "none"` / `thinking: {type: "disabled"}` / `thinkingBudget: 0`).
- `LanguageOptions.ThinkingConfig` on gemini/vertex replaces the normalized thinking mapping with a native `genai.ThinkingConfig`.

### Changed

- **Breaking**: the model binds at construction. Each provider package exposes `New(opts ...Option) *Provider` (`New(ctx, opts ...Option) (*Provider, error)` on gemini/vertex) and models derive from it by name — `p.LanguageModel("claude-sonnet-4-5")` — replacing the per-modality `New<Modality>Model` constructors. The `Model` field is removed from all request types, and the five model interfaces replace `Name() string` with `Model() string` and `Provider() string`.
- **Breaking**: gemini/vertex map `Thinking.Effort` to `thinkingLevel` (Gemini 3) instead of a token budget; Gemini 2.x models reject it — pass `BudgetTokens` for those.
- **Breaking**: openai converts a budget-only `Thinking` to the nearest `reasoning_effort` instead of silently dropping it; models without reasoning support now reject such requests.

### Fixed

- gemini/vertex `SpeechResponse.Model` falls back to the bound model when the API omits `modelVersion`, instead of coming back empty.

## [0.1.0] - 2026-07-08

Initial release.

### Added

- Five canonical model interfaces: `LanguageModel`, `EmbeddingModel`, `ImageModel`, `SpeechModel`, `TranscriptionModel`.
- Four protocol providers: `anthropic`, `openai`, `gemini`, and `vertex`.
- Language surface: streaming and non-streaming calls, tool use, thinking/reasoning, prompt caching, structured output, and usage accounting.

[Unreleased]: https://github.com/lf4096/koine/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/lf4096/koine/releases/tag/v0.3.0
[0.2.1]: https://github.com/lf4096/koine/releases/tag/v0.2.1
[0.2.0]: https://github.com/lf4096/koine/releases/tag/v0.2.0
[0.1.0]: https://github.com/lf4096/koine/releases/tag/v0.1.0
