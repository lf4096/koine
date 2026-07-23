package anthropic

import (
	"encoding/base64"
	"encoding/json"
	"fmt"

	"github.com/anthropics/anthropic-sdk-go"

	"github.com/lf4096/koine"
)

// defaultMaxTokens applies when the request leaves MaxTokens unset; the
// Messages API requires it.
const defaultMaxTokens = 4096

var effortBudgets = map[koine.ThinkingEffort]int64{
	koine.ThinkingMinimal: 1024,
	koine.ThinkingLow:     4096,
	koine.ThinkingMedium:  8192,
	koine.ThinkingHigh:    16384,
	koine.ThinkingXHigh:   24576,
	koine.ThinkingMax:     32768,
}

func encodeRequest(req *koine.LanguageRequest) (anthropic.MessageNewParams, error) {
	params := anthropic.MessageNewParams{
		Model:         anthropic.Model(req.Model),
		MaxTokens:     int64(req.MaxTokens),
		StopSequences: req.StopSequences,
	}
	if params.MaxTokens == 0 {
		params.MaxTokens = defaultMaxTokens
	}
	if req.System != "" {
		params.System = []anthropic.TextBlockParam{{Text: req.System}}
	}
	if req.Temperature != nil {
		params.Temperature = anthropic.Float(*req.Temperature)
	}
	if req.TopP != nil {
		params.TopP = anthropic.Float(*req.TopP)
	}
	if req.Thinking != nil {
		if req.Thinking.Effort == koine.ThinkingNone {
			if req.Thinking.BudgetTokens > 0 {
				return anthropic.MessageNewParams{}, fmt.Errorf("ThinkingNone cannot be combined with BudgetTokens")
			}
			disabled := anthropic.NewThinkingConfigDisabledParam()
			params.Thinking = anthropic.ThinkingConfigParamUnion{OfDisabled: &disabled}
		} else {
			budget := int64(req.Thinking.BudgetTokens)
			if budget == 0 {
				budget = effortBudgets[req.Thinking.Effort]
			}
			if budget > 0 {
				params.Thinking = anthropic.ThinkingConfigParamOfEnabled(budget)
				if params.MaxTokens <= budget {
					params.MaxTokens += budget
				}
			}
		}
	}
	if rf := req.ResponseFormat; rf != nil {
		if len(rf.Schema) == 0 {
			return anthropic.MessageNewParams{}, fmt.Errorf("ResponseFormat requires a schema")
		}
		params.OutputConfig = anthropic.OutputConfigParam{
			Format: anthropic.JSONOutputFormatParam{Schema: rf.Schema},
		}
	}
	switch req.CacheRetention {
	case koine.CacheShort:
		params.CacheControl = anthropic.NewCacheControlEphemeralParam()
	case koine.CacheLong:
		cc := anthropic.NewCacheControlEphemeralParam()
		cc.TTL = anthropic.CacheControlEphemeralTTLTTL1h
		params.CacheControl = cc
	}
	for _, t := range req.Tools {
		tool, err := encodeTool(t)
		if err != nil {
			return anthropic.MessageNewParams{}, err
		}
		params.Tools = append(params.Tools, tool)
	}
	if req.ToolChoice != nil {
		params.ToolChoice = encodeToolChoice(req.ToolChoice)
	}
	for _, m := range req.Messages {
		mp, ok, err := encodeMessage(m)
		if err != nil {
			return anthropic.MessageNewParams{}, err
		}
		if ok {
			params.Messages = append(params.Messages, mp)
		}
	}
	if o, ok := req.ProviderOptions[Name].(LanguageOptions); ok {
		if o.TopK != nil {
			params.TopK = anthropic.Int(int64(*o.TopK))
		}
	}
	return params, nil
}

// encodeTool splits the schema into the SDK's structured fields: properties
// and required map directly, everything else rides ExtraFields.
func encodeTool(t koine.Tool) (anthropic.ToolUnionParam, error) {
	var schemaParam anthropic.ToolInputSchemaParam
	extras := map[string]any{}
	for k, v := range t.InputSchema {
		switch k {
		case "properties":
			schemaParam.Properties = v
		case "required":
			required, err := stringSlice(v)
			if err != nil {
				return anthropic.ToolUnionParam{}, fmt.Errorf("tool %q schema required: %w", t.Name, err)
			}
			schemaParam.Required = required
		case "type":
		default:
			extras[k] = v
		}
	}
	if len(extras) > 0 {
		schemaParam.ExtraFields = extras
	}
	tool := anthropic.ToolUnionParamOfTool(schemaParam, t.Name)
	if t.Description != "" {
		tool.OfTool.Description = anthropic.String(t.Description)
	}
	return tool, nil
}

func stringSlice(v any) ([]string, error) {
	switch list := v.(type) {
	case []string:
		return list, nil
	case []any:
		out := make([]string, len(list))
		for i, e := range list {
			s, ok := e.(string)
			if !ok {
				return nil, fmt.Errorf("element %d is %T, want string", i, e)
			}
			out[i] = s
		}
		return out, nil
	default:
		return nil, fmt.Errorf("got %T, want string list", v)
	}
}

func encodeToolChoice(tc *koine.ToolChoice) anthropic.ToolChoiceUnionParam {
	switch tc.Mode {
	case koine.ToolChoiceNone:
		return anthropic.ToolChoiceUnionParam{OfNone: &anthropic.ToolChoiceNoneParam{}}
	case koine.ToolChoiceRequired:
		return anthropic.ToolChoiceUnionParam{OfAny: &anthropic.ToolChoiceAnyParam{}}
	case koine.ToolChoiceTool:
		return anthropic.ToolChoiceParamOfTool(tc.Name)
	default:
		return anthropic.ToolChoiceUnionParam{OfAuto: &anthropic.ToolChoiceAutoParam{}}
	}
}

// encodeMessage returns ok=false when every block was dropped (e.g. a foreign
// thinking-only message); the Messages API rejects empty content.
func encodeMessage(m koine.Message) (anthropic.MessageParam, bool, error) {
	role := anthropic.MessageParamRoleUser
	if m.Role == koine.RoleAssistant {
		role = anthropic.MessageParamRoleAssistant
	}
	var blocks []anthropic.ContentBlockParamUnion
	for _, b := range m.Blocks {
		cb, ok, err := encodeBlock(b)
		if err != nil {
			return anthropic.MessageParam{}, false, err
		}
		if ok {
			blocks = append(blocks, cb)
		}
	}
	if len(blocks) == 0 {
		return anthropic.MessageParam{}, false, nil
	}
	return anthropic.MessageParam{Role: role, Content: blocks}, true, nil
}

func encodeBlock(b koine.Block) (anthropic.ContentBlockParamUnion, bool, error) {
	switch blk := b.(type) {
	case *koine.TextBlock:
		return anthropic.NewTextBlock(blk.Text), true, nil
	case *koine.ThinkingBlock:
		if u, ok := blockFromRaw(blk.Raw); ok {
			return u, true, nil
		}
		if blk.Signature != "" && !blk.Redacted {
			return anthropic.NewThinkingBlock(blk.Signature, blk.Text), true, nil
		}
		// Foreign or unsigned thinking cannot be replayed; drop it.
		return anthropic.ContentBlockParamUnion{}, false, nil
	case *koine.ToolUseBlock:
		if u, ok := blockFromRaw(blk.Raw); ok {
			return u, true, nil
		}
		input := blk.Input
		if len(input) == 0 {
			input = json.RawMessage("{}")
		}
		return anthropic.NewToolUseBlock(blk.ID, input, blk.Name), true, nil
	case *koine.ToolResultBlock:
		return encodeToolResult(blk)
	case *koine.ImageBlock:
		img := encodeImage(blk)
		return anthropic.ContentBlockParamUnion{OfImage: &img}, true, nil
	default:
		return anthropic.ContentBlockParamUnion{}, false, fmt.Errorf("unsupported block type %q", b.BlockType())
	}
}

func blockFromRaw(raw *koine.ProviderRaw) (anthropic.ContentBlockParamUnion, bool) {
	if raw == nil || raw.Provider != Name {
		return anthropic.ContentBlockParamUnion{}, false
	}
	var u anthropic.ContentBlockParamUnion
	if err := u.UnmarshalJSON(raw.JSON); err != nil {
		return anthropic.ContentBlockParamUnion{}, false
	}
	return u, true
}

func encodeToolResult(blk *koine.ToolResultBlock) (anthropic.ContentBlockParamUnion, bool, error) {
	p := anthropic.ToolResultBlockParam{ToolUseID: blk.ToolUseID}
	if blk.IsError {
		p.IsError = anthropic.Bool(true)
	}
	if len(blk.Result) > 0 {
		if len(blk.Content) > 0 {
			return anthropic.ContentBlockParamUnion{}, false, fmt.Errorf("tool result %q sets both Content and Result", blk.ToolUseID)
		}
		p.Content = append(p.Content, anthropic.ToolResultBlockParamContentUnion{
			OfText: &anthropic.TextBlockParam{Text: string(blk.Result)},
		})
		return anthropic.ContentBlockParamUnion{OfToolResult: &p}, true, nil
	}
	for _, c := range blk.Content {
		switch cb := c.(type) {
		case *koine.TextBlock:
			p.Content = append(p.Content, anthropic.ToolResultBlockParamContentUnion{
				OfText: &anthropic.TextBlockParam{Text: cb.Text},
			})
		case *koine.ImageBlock:
			img := encodeImage(cb)
			p.Content = append(p.Content, anthropic.ToolResultBlockParamContentUnion{OfImage: &img})
		default:
			return anthropic.ContentBlockParamUnion{}, false, fmt.Errorf("unsupported tool result content %q", c.BlockType())
		}
	}
	return anthropic.ContentBlockParamUnion{OfToolResult: &p}, true, nil
}

func encodeImage(b *koine.ImageBlock) anthropic.ImageBlockParam {
	if b.URL != "" {
		return anthropic.ImageBlockParam{Source: anthropic.ImageBlockParamSourceUnion{
			OfURL: &anthropic.URLImageSourceParam{URL: b.URL},
		}}
	}
	return anthropic.ImageBlockParam{Source: anthropic.ImageBlockParamSourceUnion{
		OfBase64: &anthropic.Base64ImageSourceParam{
			MediaType: anthropic.Base64ImageSourceMediaType(b.MIMEType),
			Data:      base64.StdEncoding.EncodeToString(b.Data),
		},
	}}
}

func decodeResponse(msg *anthropic.Message) *koine.LanguageResponse {
	out := koine.Message{Role: koine.RoleAssistant}
	for _, cb := range msg.Content {
		raw := &koine.ProviderRaw{Provider: Name, JSON: json.RawMessage(cb.RawJSON())}
		switch cb.Type {
		case "text":
			out.Blocks = append(out.Blocks, &koine.TextBlock{Text: cb.Text})
		case "thinking":
			out.Blocks = append(out.Blocks, &koine.ThinkingBlock{Text: cb.Thinking, Signature: cb.Signature, Raw: raw})
		case "redacted_thinking":
			out.Blocks = append(out.Blocks, &koine.ThinkingBlock{Redacted: true, Raw: raw})
		case "tool_use":
			out.Blocks = append(out.Blocks, &koine.ToolUseBlock{ID: cb.ID, Name: cb.Name, Input: cb.Input, Raw: raw})
		}
	}
	return &koine.LanguageResponse{
		Message:    out,
		StopReason: mapStopReason(msg.StopReason),
		Usage: koine.Usage{
			InputTokens:      int(msg.Usage.InputTokens),
			OutputTokens:     int(msg.Usage.OutputTokens),
			CacheReadTokens:  int(msg.Usage.CacheReadInputTokens),
			CacheWriteTokens: int(msg.Usage.CacheCreationInputTokens),
			ReasoningTokens:  int(msg.Usage.OutputTokensDetails.ThinkingTokens),
		},
		Model:    string(msg.Model),
		Provider: Name,
	}
}

func mapStopReason(r anthropic.StopReason) koine.StopReason {
	switch r {
	case anthropic.StopReasonEndTurn, anthropic.StopReasonPauseTurn:
		return koine.StopEndTurn
	case anthropic.StopReasonToolUse:
		return koine.StopToolUse
	case anthropic.StopReasonMaxTokens:
		return koine.StopMaxTokens
	case anthropic.StopReasonStopSequence:
		return koine.StopStopSequence
	case anthropic.StopReasonRefusal:
		return koine.StopContentFilter
	default:
		return koine.StopOther
	}
}
