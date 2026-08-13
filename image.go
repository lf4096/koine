package koine

import "context"

// ImageModel generates images through a provider's dedicated image endpoint.
// Chat-native image output stays on LanguageModel as ImageBlock content in
// the response message.
type ImageModel interface {
	Model() string
	Provider() string
	GenerateImage(ctx context.Context, req *ImageRequest) (*ImageResponse, error)
}

// ImageRequest is one image-generation call.
type ImageRequest struct {
	Prompt string
	// Images are input images: empty means text-to-image; one or more routes
	// the call to the provider's edit endpoint.
	Images []*ImageBlock
	// Mask marks the region to edit for inpainting (transparent = edit here).
	Mask *ImageBlock
	// N is the number of images to generate; values <= 0 mean 1, so a
	// provider's own default batch size never leaks through.
	N int
	// Size is an absolute pixel size like "1024x1024".
	Size string
	// AspectRatio is a ratio like "16:9". Set whichever axis the target model
	// understands; the other is ignored.
	AspectRatio string
	// Seed requests reproducible output where supported; providers without a
	// seed ignore it.
	Seed            *int
	ProviderOptions map[string]any
}

// ImageResponse is the result of one call. len(Results) may be less than N
// when the provider drops filtered images.
type ImageResponse struct {
	Results []*ImageResult `json:"results"`
	Usage   Usage          `json:"usage"`
	// Model is the bound model that generated the images.
	Model    string `json:"model,omitempty"`
	Provider string `json:"provider,omitempty"`
}

// ImageResult is one generated image plus its per-image metadata.
type ImageResult struct {
	// Image is the generated content; nil when Filtered.
	Image *ImageBlock `json:"image,omitempty"`
	// RevisedPrompt is the provider's rewritten prompt, when returned.
	RevisedPrompt string `json:"revised_prompt,omitempty"`
	// Filtered reports that the provider suppressed this image for safety.
	Filtered     bool   `json:"filtered,omitempty"`
	FilterReason string `json:"filter_reason,omitempty"`
	// Raw preserves provider-native per-image JSON (safety scores etc.).
	Raw *ProviderRaw `json:"raw,omitempty"`
}
