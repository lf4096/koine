package koine

import "encoding/json"

// ProviderRaw preserves a value's provider-native JSON. When the value goes
// back to the provider that produced it, that provider resends this verbatim
// so opaque fields (thinking signatures, tool ids) survive untouched; other
// providers ignore it and convert lossily from the canonical fields.
type ProviderRaw struct {
	Provider string          `json:"provider"`
	JSON     json.RawMessage `json:"json"`
}

// ImageBlock is image content, either inline bytes or a URL (set one). It
// appears as message content on language models and as input and output on
// image models. Raw is set when the provider attaches opaque metadata to an
// image (Gemini thought signatures ride on image parts).
type ImageBlock struct {
	MIMEType string       `json:"mime_type,omitempty"`
	Data     []byte       `json:"data,omitempty"`
	URL      string       `json:"url,omitempty"`
	Raw      *ProviderRaw `json:"raw,omitempty"`
}

func (*ImageBlock) BlockType() BlockType { return BlockImage }

// Usage is normalized token accounting for one call. InputTokens excludes
// cache reads and writes on every provider; the cache fields separate read
// hits from writes because their costs differ.
type Usage struct {
	InputTokens      int `json:"input_tokens"`
	OutputTokens     int `json:"output_tokens"`
	CacheReadTokens  int `json:"cache_read_tokens,omitempty"`
	CacheWriteTokens int `json:"cache_write_tokens,omitempty"`
	ReasoningTokens  int `json:"reasoning_tokens,omitempty"`
}
