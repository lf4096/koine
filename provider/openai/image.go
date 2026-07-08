package openai

import (
	"bytes"
	"context"
	"encoding/base64"
	"fmt"
	"io"

	"github.com/openai/openai-go/v3"

	"github.com/lf4096/koine"
)

// ImageOptions carries OpenAI-specific image parameters. Pass via
// ImageRequest.ProviderOptions[openai.Name].
type ImageOptions struct {
	Quality           string // "low", "medium", "high" (gpt-image), "standard", "hd" (dall-e-3)
	Style             string // dall-e-3: "vivid", "natural"
	Background        string // gpt-image: "transparent", "opaque", "auto"
	OutputFormat      string // gpt-image: "png", "jpeg", "webp"
	OutputCompression *int
	Moderation        string // gpt-image: "low", "auto"
	// ResponseFormat "b64_json" forces inline bytes on dall-e models, whose
	// default is a URL. gpt-image models always return bytes and reject it.
	ResponseFormat string
	User           string
}

// ImageModel generates and edits images via the OpenAI images endpoints.
// (The SDK's openai.ImageModel is an unrelated model-id string type.)
type ImageModel struct {
	client openai.Client
}

// NewImageModel builds the image model; options as NewLanguageModel.
func NewImageModel(opts ...Option) *ImageModel {
	return &ImageModel{client: newClient(opts)}
}

func (m *ImageModel) Name() string { return Name }

// GenerateImage calls images/generations, or images/edits when input Images
// are present. ImageRequest.AspectRatio and Seed have no OpenAI equivalent
// and are ignored.
func (m *ImageModel) GenerateImage(ctx context.Context, req *koine.ImageRequest) (*koine.ImageResponse, error) {
	o, _ := req.ProviderOptions[Name].(ImageOptions)
	n := int64(req.N)
	if n <= 0 {
		n = 1
	}
	var (
		resp *openai.ImagesResponse
		err  error
	)
	if len(req.Images) > 0 {
		resp, err = m.edit(ctx, req, o, n)
	} else {
		params := openai.ImageGenerateParams{
			Prompt:         req.Prompt,
			Model:          openai.ImageModel(req.Model),
			N:              openai.Int(n),
			Size:           openai.ImageGenerateParamsSize(req.Size),
			Quality:        openai.ImageGenerateParamsQuality(o.Quality),
			Style:          openai.ImageGenerateParamsStyle(o.Style),
			Background:     openai.ImageGenerateParamsBackground(o.Background),
			OutputFormat:   openai.ImageGenerateParamsOutputFormat(o.OutputFormat),
			Moderation:     openai.ImageGenerateParamsModeration(o.Moderation),
			ResponseFormat: openai.ImageGenerateParamsResponseFormat(o.ResponseFormat),
		}
		if o.OutputCompression != nil {
			params.OutputCompression = openai.Int(int64(*o.OutputCompression))
		}
		if o.User != "" {
			params.User = openai.String(o.User)
		}
		resp, err = m.client.Images.Generate(ctx, params)
	}
	if err != nil {
		return nil, wrapErr(err)
	}
	return decodeImages(resp, req.Model)
}

func (m *ImageModel) edit(ctx context.Context, req *koine.ImageRequest, o ImageOptions, n int64) (*openai.ImagesResponse, error) {
	var files []io.Reader
	for i, img := range req.Images {
		if len(img.Data) == 0 {
			return nil, fmt.Errorf("edit input image %d has no inline data", i)
		}
		files = append(files, openai.File(bytes.NewReader(img.Data), "image."+extFromMIME(img.MIMEType), img.MIMEType))
	}
	params := openai.ImageEditParams{
		Image:  openai.ImageEditParamsImageUnion{OfFileArray: files},
		Prompt: req.Prompt,
		Model:  openai.ImageModel(req.Model),
		N:      openai.Int(n),
		Size:   openai.ImageEditParamsSize(req.Size),
	}
	if req.Mask != nil {
		params.Mask = openai.File(bytes.NewReader(req.Mask.Data), "mask."+extFromMIME(req.Mask.MIMEType), req.Mask.MIMEType)
	}
	if o.User != "" {
		params.User = openai.String(o.User)
	}
	return m.client.Images.Edit(ctx, params)
}

func decodeImages(resp *openai.ImagesResponse, model string) (*koine.ImageResponse, error) {
	out := &koine.ImageResponse{
		Usage: koine.Usage{
			InputTokens:  int(resp.Usage.InputTokens),
			OutputTokens: int(resp.Usage.OutputTokens),
		},
		Model:    model,
		Provider: Name,
	}
	mime := "image/png"
	if resp.OutputFormat != "" {
		mime = "image/" + string(resp.OutputFormat)
	}
	for _, img := range resp.Data {
		result := &koine.ImageResult{RevisedPrompt: img.RevisedPrompt}
		switch {
		case img.B64JSON != "":
			data, err := base64.StdEncoding.DecodeString(img.B64JSON)
			if err != nil {
				return nil, wrapErr(fmt.Errorf("decode image: %w", err))
			}
			result.Image = &koine.ImageBlock{MIMEType: mime, Data: data}
		case img.URL != "":
			result.Image = &koine.ImageBlock{URL: img.URL}
		}
		out.Results = append(out.Results, result)
	}
	return out, nil
}

func extFromMIME(mime string) string {
	switch mime {
	case "image/jpeg":
		return "jpg"
	case "image/webp":
		return "webp"
	default:
		return "png"
	}
}
