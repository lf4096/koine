# Changelog

All notable changes to this project are documented here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `ThinkingNone` disables reasoning explicitly (`reasoning_effort: "none"` / `thinking: {type: "disabled"}` / `thinkingBudget: 0`).
- `LanguageOptions.ThinkingConfig` on gemini/vertex replaces the normalized thinking mapping with a native `genai.ThinkingConfig`.

### Changed

- **Breaking**: gemini/vertex map `Thinking.Effort` to `thinkingLevel` (Gemini 3) instead of a token budget; Gemini 2.x models reject it — pass `BudgetTokens` for those.
- openai converts a budget-only `Thinking` to the nearest `reasoning_effort` instead of silently dropping it; models without reasoning support now reject such requests.

## [0.1.0] - 2026-07-08

Initial release.

### Added

- Five canonical model interfaces: `LanguageModel`, `EmbeddingModel`, `ImageModel`, `SpeechModel`, `TranscriptionModel`.
- Four protocol providers: `anthropic`, `openai`, `gemini`, and `vertex`.
- Language surface: streaming and non-streaming calls, tool use, thinking/reasoning, prompt caching, structured output, and usage accounting.

[0.1.0]: https://github.com/lf4096/koine/releases/tag/v0.1.0
