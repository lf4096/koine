# Changelog

All notable changes to this project are documented here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-07-08

Initial release.

### Added

- Five canonical model interfaces: `LanguageModel`, `EmbeddingModel`, `ImageModel`, `SpeechModel`, `TranscriptionModel`.
- Four protocol providers: `anthropic`, `openai`, `gemini`, and `vertex`.
- Language surface: streaming and non-streaming calls, tool use, thinking/reasoning, prompt caching, structured output, and usage accounting.

[0.1.0]: https://github.com/lf4096/koine/releases/tag/v0.1.0
