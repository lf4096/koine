package vertex

import (
	"context"

	"github.com/lf4096/koine"
	"github.com/lf4096/koine/internal/genaiengine"
)

// ImageOptions carries Imagen-specific image parameters.
type ImageOptions = genaiengine.ImageOptions

// ImageModel generates images via Imagen models on Vertex.
type ImageModel struct {
	e genaiengine.Engine
}

var _ koine.ImageModel = (*ImageModel)(nil)

// ImageModel builds the image model.
func (p *Provider) ImageModel(model string) *ImageModel {
	return &ImageModel{e: p.engine(model)}
}

func (m *ImageModel) Model() string { return m.e.Model }

func (m *ImageModel) Provider() string { return Name }

// GenerateImage performs one Imagen predict call.
func (m *ImageModel) GenerateImage(ctx context.Context, req *koine.ImageRequest) (*koine.ImageResponse, error) {
	return m.e.GenerateImage(ctx, req)
}
