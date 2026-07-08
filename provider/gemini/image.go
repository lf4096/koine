package gemini

import (
	"context"

	"github.com/lf4096/koine"
	"github.com/lf4096/koine/internal/genaiengine"
)

// ImageOptions carries Imagen-specific image parameters.
type ImageOptions = genaiengine.ImageOptions

// ImageModel generates images via Imagen models.
type ImageModel struct {
	e genaiengine.Engine
}

// NewImageModel builds the image model; options as NewLanguageModel.
func NewImageModel(ctx context.Context, opts ...Option) (*ImageModel, error) {
	e, err := newEngine(ctx, opts)
	if err != nil {
		return nil, err
	}
	return &ImageModel{e: e}, nil
}

func (m *ImageModel) Name() string { return Name }

// GenerateImage performs one Imagen generateImages call.
func (m *ImageModel) GenerateImage(ctx context.Context, req *koine.ImageRequest) (*koine.ImageResponse, error) {
	return m.e.GenerateImage(ctx, req)
}
