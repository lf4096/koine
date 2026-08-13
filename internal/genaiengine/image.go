package genaiengine

import (
	"context"
	"encoding/json"

	"google.golang.org/genai"

	"github.com/lf4096/koine"
)

// ImageOptions carries Imagen parameters specific to genai-based providers.
// The SDK rejects some fields per backend (NegativePrompt, GuidanceScale, and
// EnhancePrompt are Vertex-only).
type ImageOptions struct {
	NegativePrompt    string
	GuidanceScale     *float32
	SafetyFilterLevel genai.SafetyFilterLevel
	PersonGeneration  genai.PersonGeneration
	IncludeRAIReason  bool
	OutputMIMEType    string
	EnhancePrompt     bool
}

// GenerateImage performs one Imagen generateImages call.
func (e *Engine) GenerateImage(ctx context.Context, req *koine.ImageRequest) (*koine.ImageResponse, error) {
	if len(req.Images) > 0 || req.Mask != nil {
		return nil, &koine.Error{Provider: e.Name, Message: "image editing is not supported; Imagen edit APIs are Vertex-specific and not covered yet"}
	}
	n := req.N
	if n <= 0 {
		n = 1
	}
	config := &genai.GenerateImagesConfig{
		NumberOfImages: int32(n),
		AspectRatio:    req.AspectRatio,
	}
	if req.Seed != nil {
		config.Seed = new(int32(*req.Seed))
	}
	if o, ok := req.ProviderOptions[e.Name].(ImageOptions); ok {
		config.NegativePrompt = o.NegativePrompt
		config.GuidanceScale = o.GuidanceScale
		config.SafetyFilterLevel = o.SafetyFilterLevel
		config.PersonGeneration = o.PersonGeneration
		config.IncludeRAIReason = o.IncludeRAIReason
		config.OutputMIMEType = o.OutputMIMEType
		config.EnhancePrompt = o.EnhancePrompt
	}
	resp, err := e.Client.Models.GenerateImages(ctx, e.Model, req.Prompt, config)
	if err != nil {
		return nil, e.wrapErr(err)
	}
	out := &koine.ImageResponse{Model: e.Model, Provider: e.Name}
	for _, gi := range resp.GeneratedImages {
		result := &koine.ImageResult{
			RevisedPrompt: gi.EnhancedPrompt,
			Filtered:      gi.RAIFilteredReason != "",
			FilterReason:  gi.RAIFilteredReason,
		}
		// The SDK builds an empty Image object even for filtered predictions.
		if gi.Image != nil && (len(gi.Image.ImageBytes) > 0 || gi.Image.GCSURI != "") {
			result.Image = &koine.ImageBlock{
				MIMEType: gi.Image.MIMEType,
				Data:     gi.Image.ImageBytes,
				URL:      gi.Image.GCSURI,
			}
		}
		if gi.SafetyAttributes != nil {
			if data, err := json.Marshal(gi.SafetyAttributes); err == nil {
				result.Raw = &koine.ProviderRaw{Provider: e.Name, JSON: data}
			}
		}
		out.Results = append(out.Results, result)
	}
	return out, nil
}
