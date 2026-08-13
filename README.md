# koine

> **koine** /kɔɪˈneɪ/ — the common Greek tongue of the Hellenistic world. One tongue for every LLM.

koine is a thin, fidelity-first Go library that speaks every major LLM API through one canonical interface per model type: language, embedding, image, speech, transcription. It covers OpenAI (and every OpenAI-compatible endpoint), Anthropic Claude, Gemini, and Vertex AI, implemented on top of the official provider SDKs, translating formats and nothing else. Everything above the model call (the agent loop, tool execution, retries, model catalogs) stays with you.

```
go get github.com/lf4096/koine
```

## Models and providers

Five model interfaces, four protocol providers. Each modality is its own narrow interface, implemented only where the provider actually has the endpoint:

| | `openai` | `gemini` | `vertex` | `anthropic` |
|---|---|---|---|---|
| `LanguageModel` (chat) | ✓ | ✓ | ✓ | ✓ |
| `EmbeddingModel` | ✓ | ✓ | ✓ | — |
| `ImageModel` | ✓ generate + edit | ✓ Imagen | ✓ Imagen | — |
| `SpeechModel` (TTS) | ✓ | ✓ | ✓ | — |
| `TranscriptionModel` (STT) | ✓ | — | — | — |

The four providers are protocol adapters, not per-vendor bindings, so they cover the real model landscape:

- **`anthropic`** — the Anthropic Messages protocol (Claude).
- **`openai`** — the OpenAI protocol: OpenAI itself plus every OpenAI-compatible endpoint (DeepSeek, Kimi, Qwen, GLM, ...), including their `reasoning_content` thinking extension.
- **`gemini`** — the Gemini API.
- **`vertex`** — the same wire format as gemini, spoken through Vertex AI paths and auth (GCP credentials, express API keys, or self-authenticating gateways). Raw blocks replay across gemini and vertex without loss.

Construction is two-layered: `New(...)` in each provider package builds a provider holding credentials, endpoint, and one SDK client, and models derive from it by name — `openai.New().LanguageModel("gpt-4.1-mini")`, or `p, err := vertex.New(ctx, vertex.WithProject("my-project"), vertex.WithLocation("global"))` then `p.ImageModel("imagen-4.0-generate-001")`. gemini and vertex take a `ctx` and return an error because their SDK builds the client eagerly; deriving a model never fails. Without options, credentials come from the environment exactly as the official SDK defines (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`, Application Default Credentials).

## Why koine

Most unified LLM libraries normalize providers down to a common denominator, and the first things lost are exactly what agent runtimes need most:

- **Thinking fidelity.** Anthropic thinking blocks must round-trip with their signatures byte-for-byte or the API rejects the request. Gemini attaches opaque thought signatures to parts. koine stores each block's provider-native JSON (`ProviderRaw`) and resends it verbatim when the conversation continues on the same provider.
- **Prompt caching.** Cache hits change long-conversation cost by roughly 10x. koine exposes caching as a first-class request knob (`CacheRetention`) instead of hiding it.
- **Narrow beats broad.** Each provider only translates formats; transport, auth, and retries stay in the official provider SDK underneath.

## Quickstart

```go
package main

import (
	"context"
	"fmt"

	"github.com/lf4096/koine"
	"github.com/lf4096/koine/provider/anthropic"
)

func main() {
	m := anthropic.New().LanguageModel("claude-sonnet-4-5") // reads ANTHROPIC_API_KEY

	resp, err := m.Complete(context.Background(), &koine.LanguageRequest{
		Messages: []koine.Message{koine.UserText("Say hello in Greek.")},
	})
	if err != nil {
		panic(err)
	}
	fmt.Println(resp.Message.Text())
}
```

## Language models

`LanguageModel` is the chat surface an agent loop consumes: one call with tools, thinking, structured output, prompt caching, and usage accounting, expressed in a canonical block-style message model.

```go
type LanguageModel interface {
	Model() string
	Provider() string
	Capabilities() LanguageCapabilities
	Complete(ctx context.Context, req *LanguageRequest) (*LanguageResponse, error)
	Stream(ctx context.Context, req *LanguageRequest) (*LanguageStream, error)
}
```

### Streaming

```go
stream, err := m.Stream(ctx, req)
if err != nil {
	return err
}
defer stream.Close()
for ev := range stream.Events() {
	switch ev.Type {
	case koine.EventTextDelta:
		fmt.Print(ev.Text)
	case koine.EventThinkingDelta:
		// render reasoning...
	}
}
if err := stream.Err(); err != nil {
	return err
}
final := stream.Response() // accumulated message, usage, stop reason
```

`Events()` is a standard Go iterator; the scanner form (`for stream.Next() { ev := stream.Event() }`) remains for callers that need it. One event set across all providers: `text_delta`, `thinking_delta`, `tool_call_start`, `tool_call_delta`, `tool_call_end`, `usage`, `stop`. Events carry the content-block ordinal in `Index`; a change of index is a block boundary. Errors surface through `stream.Err()`, Go-style, never as in-band events. `stream.Collect()` drains the remainder and returns the final response in one call — `Complete` is exactly `Stream` + `Collect`.

### The agent loop

koine stops at the single model call; the loop is yours. A complete tool round trip:

```go
req := &koine.LanguageRequest{
	Tools: []koine.Tool{{
		Name:        "get_weather",
		Description: "Get current weather for a city",
		InputSchema: map[string]any{
			"type":       "object",
			"properties": map[string]any{"city": map[string]any{"type": "string"}},
			"required":   []string{"city"},
		},
	}},
	Messages: []koine.Message{koine.UserText("Weather in Paris?")},
}

for {
	resp, err := m.Complete(ctx, req)
	if err != nil {
		return err
	}
	if resp.StopReason != koine.StopToolUse {
		fmt.Println(resp.Message.Text())
		return nil
	}
	req.Messages = append(req.Messages, resp.Message)
	for _, call := range resp.Message.ToolUses() {
		result := runTool(call.Name, call.Input)
		req.Messages = append(req.Messages, koine.ToolResultText(call.ID, result, false))
	}
}
```

Tools that return JSON should answer with `koine.ToolResultJSON(call.ID, resultJSON, false)` instead: providers whose protocol takes structured results (Gemini) receive the object natively, the rest receive the same bytes as text.

Appending `resp.Message` back into the conversation is the fidelity path: its blocks carry `ProviderRaw`, so thinking signatures and tool ids return to the provider untouched.

### Message model

Messages are block lists, Anthropic-style — the most agent-complete shape of the three protocols, and the easiest to convert downward:

| Block | Fields | Notes |
|---|---|---|
| `TextBlock` | `Text`, `Raw` | `Raw` set when the provider attaches metadata to text (Gemini thought signatures) |
| `ThinkingBlock` | `Text`, `Signature`, `Redacted`, `Raw` | replayable only via `Raw` on the same provider; dropped cross-provider |
| `ToolUseBlock` | `ID`, `Name`, `Input`, `Raw` | `Input` is raw JSON as the model produced it — validate before use |
| `ToolResultBlock` | `ToolUseID`, `Content`, `Result`, `IsError` | lives in `RoleTool` messages; `Result` carries a structured JSON payload, delivered as a native object where the protocol takes one; `Content` images ride only where the protocol accepts them (Anthropic) — others error |
| `ImageBlock` | `MIMEType`, `Data`, `URL` | inline bytes or URL |

Roles are `user`, `assistant`, and `tool`. Providers reshape the tool role to each protocol's native form: Anthropic merges tool messages into user turns with `tool_result` blocks, OpenAI fans them out to `role:"tool"` messages, Gemini converts them to `functionResponse` parts.

The whole model marshals to JSON with type discriminators, so a `[]koine.Message` is also a session persistence format.

### Thinking

```go
req.Thinking = &koine.Thinking{Effort: koine.ThinkingHigh}
```

One normalized effort scale (`none` / `minimal` / `low` / `medium` / `high` / `xhigh` / `max`) plus a token budget. Each provider sends its native dial and converts the other: `reasoning_effort` on OpenAI, a token budget on Anthropic, `thinkingLevel` or `thinkingBudget` on Gemini. On Gemini, `Effort` maps to `thinkingLevel`, which needs Gemini 3; pass `BudgetTokens` for Gemini 2.x models, which reject it. `ThinkingNone` disables reasoning; models that cannot switch it off reject the request. For full control of Gemini's native knobs, pass `gemini.LanguageOptions{ThinkingConfig: ...}` (or `vertex.LanguageOptions`, the same type), which replaces the normalized mapping.

### Structured output

```go
req.ResponseFormat = &koine.ResponseFormat{
	Schema: map[string]any{
		"type":       "object",
		"properties": map[string]any{"capital": map[string]any{"type": "string"}},
		"required":   []string{"capital"},
	},
}
```

Maps to each provider's native mechanism: OpenAI `response_format: json_schema` (set `Strict: true` for enforced adherence — schema subset rules apply), Anthropic `output_config`, Gemini `responseJsonSchema`. An empty `Schema` requests schemaless JSON mode where the provider has one. Providers that cannot express the constraint return an error instead of silently dropping it; check `Capabilities().StructuredOutput` to fall back to a forced tool call (`ToolChoice` + a tool whose `InputSchema` is your output schema). One protocol constraint to know: Gemini rejects `ResponseFormat` combined with `Tools`.

### Prompt caching

```go
req.CacheRetention = koine.CacheShort // Anthropic: 5m TTL; koine.CacheLong: 1h
```

On Anthropic this sets a top-level `cache_control`, caching the whole prefix up to the last block. OpenAI and Gemini cache automatically and ignore the knob. Cache accounting is always visible: `Usage.CacheReadTokens` and `Usage.CacheWriteTokens`.

### Capabilities

`Capabilities()` reports what a provider can express (`Thinking`, `CacheControl`, `ParallelToolCalls`, `Images`, `StructuredOutput`), so a loop can degrade before sending a request the provider would reject.

## Embedding models

```go
em := openai.New().EmbeddingModel("text-embedding-3-small")

resp, err := em.Embed(ctx, &koine.EmbedRequest{
	Inputs: []string{"first text", "second text"},
	Task:   koine.EmbedTaskQuery, // normalized task hint; mapped or ignored per provider
})
// resp.Embeddings: one []float32 per input, in input order
```

Embeddings are `[]float32` end to end — the native output type of the Gemini SDK and the storage type of mainstream vector stores. `Dimensions` truncates the output vector where the model supports it; `Task` (`query` / `document` / `similarity` / `classification` / `clustering`) tunes output on providers with the concept and is ignored elsewhere. `Usage` reports input tokens where the provider bills them.

## Image models

```go
im := openai.New().ImageModel("gpt-image-1")

resp, err := im.GenerateImage(ctx, &koine.ImageRequest{
	Prompt: "an ancient greek scroll",
	Size:   "1024x1024", // or AspectRatio: "16:9" — set whichever the model understands
	N:      2,
})
for _, r := range resp.Results {
	if r.Filtered {
		continue // provider suppressed this image for safety; see r.FilterReason
	}
	save(r.Image.Data)
}
```

Passing input images routes the call to the provider's edit endpoint:

```go
resp, err := im.GenerateImage(ctx, &koine.ImageRequest{
	Prompt: "replace the sky with an aurora",
	Images: []*koine.ImageBlock{{MIMEType: "image/png", Data: original}},
	Mask:   &koine.ImageBlock{MIMEType: "image/png", Data: mask}, // optional; transparent = edit here
})
```

Editing is OpenAI-only today — gemini and vertex (Imagen) return an error rather than silently generating from scratch. Per-image metadata survives normalization: `RevisedPrompt`, safety filtering, and the provider's raw per-image JSON in `Raw`.

Chat-native image output (Gemini Flash Image) needs no `ImageModel` — it arrives as an `ImageBlock` in the chat response.

## Speech models (TTS)

```go
sm := openai.New().SpeechModel("gpt-4o-mini-tts")

resp, err := sm.GenerateSpeech(ctx, &koine.SpeechRequest{
	Text:   "Koine speaks every tongue.",
	Voice:  "alloy", // provider voice id
	Format: "mp3",   // provider format token; ignored where output is fixed
})
os.WriteFile("hello.mp3", resp.Audio, 0o644)
```

`resp.MIMEType` names the actual media type — Gemini TTS models ignore `Format` and always return raw PCM. `Instructions` steers delivery and `Speed` scales rate on models that accept them.

## Transcription models (STT)

```go
tm := openai.New().TranscriptionModel("gpt-4o-mini-transcribe")

resp, err := tm.Transcribe(ctx, &koine.TranscriptionRequest{
	Audio:    audioBytes,
	MIMEType: "audio/mpeg",
	Language: "en", // optional ISO-639-1 hint
})
fmt.Println(resp.Text)
```

Transcripts normalize `Segments` (timestamped spans) and `Duration` where the provider reports them, and keep the full provider payload in `Raw` for fields koine does not normalize (word timestamps, logprobs).

## Multi-tenant / BYOK

Providers are cheap to construct — build one per credential, and derive models from it:

```go
p := openai.New(
	openai.WithAPIKey(tenantKey),
	openai.WithBaseURL("https://api.deepseek.com"),
)
m := p.LanguageModel("deepseek-chat")
```

Every provider accepts `WithAPIKey`, `WithBaseURL`, `WithHTTPClient`, and `WithClient(...)` to inject a fully configured official SDK client (custom proxies, middleware, ...). `vertex` adds GCP-specific construction: `WithProject`/`WithLocation`/`WithCredentials` for ADC, `WithAPIKey` for express mode, or `WithBaseURL`+`WithAPIVersion`+`WithHeader` for bearer-token gateways that proxy Vertex paths.

## Provider options

Portable fields live on the request types; everything provider-specific rides `ProviderOptions`, keyed by provider name, typed by each provider package (`LanguageOptions`, `EmbedOptions`, `ImageOptions`):

```go
req.ProviderOptions = map[string]any{
	openai.Name: openai.LanguageOptions{
		LegacyMaxTokens: true,                                    // older compat endpoints
		ExtraBody:       map[string]any{"enable_thinking": true}, // vendor extensions
	},
}
```

## Errors

All failures normalize to `*koine.Error` with the provider name, HTTP status, provider code, and the wrapped SDK error. `koine.Retryable(err)` classifies 408/429/5xx without any type assertion:

```go
if koine.Retryable(err) {
	// back off and retry
}
```

For the full details (status, provider code), unwrap with `errors.AsType[*koine.Error](err)`.

## Testing

`go test ./...` runs fully offline against recorded provider fixtures. Live tests are skipped unless credentials are present: copy `.env.example` to `.env`, fill in the keys you have, and run again.

## Non-goals

No agent loop, no prompt templates, no model catalog or cost tables, no gateway. If you need an egress gateway (central keys, quotas, audit), point a provider's base URL at it — the abstraction is unchanged.

## License

Apache-2.0. See [LICENSE](LICENSE).
