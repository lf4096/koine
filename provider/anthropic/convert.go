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

var effortLevels = map[koine.ThinkingEffort]anthropic.OutputConfigEffort{
	koine.ThinkingMinimal: anthropic.OutputConfigEffortLow,
	koine.ThinkingLow:     anthropic.OutputConfigEffortLow,
	koine.ThinkingMedium:  anthropic.OutputConfigEffortMedium,
	koine.ThinkingHigh:    anthropic.OutputConfigEffortHigh,
	koine.ThinkingXHigh:   anthropic.OutputConfigEffortXhigh,
	koine.ThinkingMax:     anthropic.OutputConfigEffortMax,
}

// effortReserves is the room max_tokens keeps for thinking at each effort:
// adaptive thinking draws from max_tokens without a budget of its own.
var effortReserves = map[koine.ThinkingEffort]int64{
	koine.ThinkingMinimal: 1024,
	koine.ThinkingLow:     4096,
	koine.ThinkingMedium:  8192,
	koine.ThinkingHigh:    16384,
	koine.ThinkingXHigh:   24576,
	koine.ThinkingMax:     32768,
}

func encodeRequest(model string, req *koine.LanguageRequest) (anthropic.MessageNewParams, error) {
	params := anthropic.MessageNewParams{
		Model:         anthropic.Model(model),
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
	var reserve int64
	if req.Thinking != nil {
		// Requested thinking comes back readable: newer models omit the text
		// unless display asks for a summary.
		switch {
		case req.Thinking.Effort == koine.ThinkingNone:
			if req.Thinking.BudgetTokens > 0 {
				return anthropic.MessageNewParams{}, fmt.Errorf("ThinkingNone cannot be combined with BudgetTokens")
			}
			disabled := anthropic.NewThinkingConfigDisabledParam()
			params.Thinking = anthropic.ThinkingConfigParamUnion{OfDisabled: &disabled}
		case req.Thinking.BudgetTokens > 0:
			params.Thinking = anthropic.ThinkingConfigParamUnion{OfEnabled: &anthropic.ThinkingConfigEnabledParam{
				BudgetTokens: int64(req.Thinking.BudgetTokens),
				Display:      anthropic.ThinkingConfigEnabledDisplaySummarized,
			}}
		default:
			if effort, ok := effortLevels[req.Thinking.Effort]; ok {
				params.Thinking = anthropic.ThinkingConfigParamUnion{OfAdaptive: &anthropic.ThinkingConfigAdaptiveParam{
					Display: anthropic.ThinkingConfigAdaptiveDisplaySummarized,
				}}
				params.OutputConfig.Effort = effort
				reserve = effortReserves[req.Thinking.Effort]
			}
		}
	}
	o, _ := req.ProviderOptions[Name].(LanguageOptions)
	if o.Thinking != nil {
		params.Thinking = *o.Thinking
		params.OutputConfig.Effort = ""
		reserve = 0
	}
	if o.Effort != "" {
		params.OutputConfig.Effort = o.Effort
	}
	if e := params.Thinking.OfEnabled; e != nil {
		reserve = e.BudgetTokens
	}
	if params.MaxTokens <= reserve {
		params.MaxTokens += reserve
	}
	if rf := req.ResponseFormat; rf != nil {
		if len(rf.Schema) == 0 {
			return anthropic.MessageNewParams{}, fmt.Errorf("ResponseFormat requires a schema")
		}
		params.OutputConfig.Format = anthropic.JSONOutputFormatParam{Schema: rf.Schema}
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
	if o.TopK != nil {
		params.TopK = anthropic.Int(int64(*o.TopK))
	}
	placeCacheMarkers(&params, req.CacheRetention)
	return params, nil
}

func placeCacheMarkers(params *anthropic.MessageNewParams, retention koine.CacheRetention) {
	marker := anthropic.NewCacheControlEphemeralParam()
	switch retention {
	case koine.CacheShort:
	case koine.CacheLong:
		marker.TTL = anthropic.CacheControlEphemeralTTLTTL1h
	default:
		return
	}
	if n := len(params.Tools); n > 0 {
		if cc := params.Tools[n-1].GetCacheControl(); cc != nil {
			*cc = marker
		}
	}
	if n := len(params.System); n > 0 {
		params.System[n-1].CacheControl = marker
	}
	msgs := params.Messages
	if len(msgs) == 0 {
		return
	}
	markLastBlock(msgs[len(msgs)-1], marker)
	// Also mark where the previous request ended: the API finds an earlier cache
	// entry only within a limited number of blocks before each marker, so a
	// request that adds many blocks would otherwise miss it.
	i := len(msgs) - 1
	for i >= 0 && msgs[i].Role == anthropic.MessageParamRoleUser {
		i--
	}
	for i >= 0 && msgs[i].Role != anthropic.MessageParamRoleUser {
		i--
	}
	if i >= 0 {
		markLastBlock(msgs[i], marker)
	}
}

func markLastBlock(m anthropic.MessageParam, marker anthropic.CacheControlEphemeralParam) {
	for i := len(m.Content) - 1; i >= 0; i-- {
		if cc := m.Content[i].GetCacheControl(); cc != nil {
			*cc = marker
			return
		}
	}
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
	case anthropic.StopReasonMaxTokens, anthropic.StopReasonModelContextWindowExceeded:
		return koine.StopMaxTokens
	case anthropic.StopReasonStopSequence:
		return koine.StopStopSequence
	case anthropic.StopReasonRefusal:
		return koine.StopContentFilter
	default:
		return koine.StopOther
	}
}
