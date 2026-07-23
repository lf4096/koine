package openai

import (
	"encoding/base64"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/option"
	"github.com/openai/openai-go/v3/shared"

	"github.com/lf4096/koine"
)

var effortLevels = map[koine.ThinkingEffort]shared.ReasoningEffort{
	koine.ThinkingNone:    "none",
	koine.ThinkingMinimal: "minimal",
	koine.ThinkingLow:     "low",
	koine.ThinkingMedium:  "medium",
	koine.ThinkingHigh:    "high",
	koine.ThinkingXHigh:   "xhigh",
	koine.ThinkingMax:     "xhigh",
}

// The thresholds mirror the anthropic provider's effort table.
func effortForBudget(budget int) koine.ThinkingEffort {
	switch {
	case budget <= 0:
		return ""
	case budget <= 1024:
		return koine.ThinkingMinimal
	case budget <= 4096:
		return koine.ThinkingLow
	case budget <= 8192:
		return koine.ThinkingMedium
	case budget <= 16384:
		return koine.ThinkingHigh
	default:
		return koine.ThinkingXHigh
	}
}

func encodeRequest(req *koine.LanguageRequest) (openai.ChatCompletionNewParams, []option.RequestOption, error) {
	o, _ := req.ProviderOptions[Name].(LanguageOptions)
	params := openai.ChatCompletionNewParams{Model: shared.ChatModel(req.Model)}
	if req.MaxTokens > 0 {
		if o.LegacyMaxTokens {
			params.MaxTokens = openai.Int(int64(req.MaxTokens))
		} else {
			params.MaxCompletionTokens = openai.Int(int64(req.MaxTokens))
		}
	}
	if req.Temperature != nil {
		params.Temperature = openai.Float(*req.Temperature)
	}
	if req.TopP != nil {
		params.TopP = openai.Float(*req.TopP)
	}
	if len(req.StopSequences) > 0 {
		params.Stop = openai.ChatCompletionNewParamsStopUnion{OfStringArray: req.StopSequences}
	}
	if req.Thinking != nil {
		if req.Thinking.Effort == koine.ThinkingNone && req.Thinking.BudgetTokens > 0 {
			return openai.ChatCompletionNewParams{}, nil, fmt.Errorf("ThinkingNone cannot be combined with BudgetTokens")
		}
		effort := req.Thinking.Effort
		if effort == "" {
			effort = effortForBudget(req.Thinking.BudgetTokens)
		}
		params.ReasoningEffort = effortLevels[effort]
	}
	if rf := req.ResponseFormat; rf != nil {
		params.ResponseFormat = encodeResponseFormat(rf)
	}
	if !o.NoStreamUsage {
		params.StreamOptions = openai.ChatCompletionStreamOptionsParam{IncludeUsage: openai.Bool(true)}
	}
	if req.System != "" {
		params.Messages = append(params.Messages, openai.SystemMessage(req.System))
	}
	for _, m := range req.Messages {
		ms, err := encodeMessage(m)
		if err != nil {
			return openai.ChatCompletionNewParams{}, nil, err
		}
		params.Messages = append(params.Messages, ms...)
	}
	for _, t := range req.Tools {
		params.Tools = append(params.Tools, encodeTool(t))
	}
	if req.ToolChoice != nil {
		params.ToolChoice = encodeToolChoice(req.ToolChoice)
	}
	var reqOpts []option.RequestOption
	for k, v := range o.ExtraBody {
		reqOpts = append(reqOpts, option.WithJSONSet(k, v))
	}
	return params, reqOpts, nil
}

func encodeResponseFormat(rf *koine.ResponseFormat) openai.ChatCompletionNewParamsResponseFormatUnion {
	if len(rf.Schema) == 0 {
		return openai.ChatCompletionNewParamsResponseFormatUnion{
			OfJSONObject: &shared.ResponseFormatJSONObjectParam{},
		}
	}
	name := rf.Name
	if name == "" {
		name = "response"
	}
	js := shared.ResponseFormatJSONSchemaJSONSchemaParam{Name: name, Schema: rf.Schema}
	if rf.Description != "" {
		js.Description = openai.String(rf.Description)
	}
	if rf.Strict {
		js.Strict = openai.Bool(true)
	}
	return openai.ChatCompletionNewParamsResponseFormatUnion{
		OfJSONSchema: &shared.ResponseFormatJSONSchemaParam{JSONSchema: js},
	}
}

func encodeTool(t koine.Tool) openai.ChatCompletionToolUnionParam {
	fn := shared.FunctionDefinitionParam{Name: t.Name}
	if t.Description != "" {
		fn.Description = openai.String(t.Description)
	}
	if len(t.InputSchema) > 0 {
		fn.Parameters = shared.FunctionParameters(t.InputSchema)
	}
	return openai.ChatCompletionFunctionTool(fn)
}

func encodeToolChoice(tc *koine.ToolChoice) openai.ChatCompletionToolChoiceOptionUnionParam {
	switch tc.Mode {
	case koine.ToolChoiceNone:
		return openai.ChatCompletionToolChoiceOptionUnionParam{OfAuto: openai.String("none")}
	case koine.ToolChoiceRequired:
		return openai.ChatCompletionToolChoiceOptionUnionParam{OfAuto: openai.String("required")}
	case koine.ToolChoiceTool:
		return openai.ChatCompletionToolChoiceOptionUnionParam{
			OfFunctionToolChoice: &openai.ChatCompletionNamedToolChoiceParam{
				Function: openai.ChatCompletionNamedToolChoiceFunctionParam{Name: tc.Name},
			},
		}
	default:
		return openai.ChatCompletionToolChoiceOptionUnionParam{OfAuto: openai.String("auto")}
	}
}

// encodeMessage reshapes one canonical message to its Chat Completions form:
// tool messages fan out to one role:"tool" message per result, and assistant
// thinking blocks are dropped (the protocol does not accept reasoning back).
func encodeMessage(m koine.Message) ([]openai.ChatCompletionMessageParamUnion, error) {
	switch m.Role {
	case koine.RoleTool:
		var out []openai.ChatCompletionMessageParamUnion
		for _, b := range m.Blocks {
			tr, ok := b.(*koine.ToolResultBlock)
			if !ok {
				return nil, fmt.Errorf("tool message requires tool_result blocks, got %q", b.BlockType())
			}
			text, err := toolResultText(tr)
			if err != nil {
				return nil, err
			}
			out = append(out, openai.ToolMessage(text, tr.ToolUseID))
		}
		return out, nil
	case koine.RoleAssistant:
		return encodeAssistantMessage(m)
	default:
		return encodeUserMessage(m)
	}
}

func encodeAssistantMessage(m koine.Message) ([]openai.ChatCompletionMessageParamUnion, error) {
	var p openai.ChatCompletionAssistantMessageParam
	var text strings.Builder
	for _, b := range m.Blocks {
		switch blk := b.(type) {
		case *koine.TextBlock:
			text.WriteString(blk.Text)
		case *koine.ThinkingBlock:
		case *koine.ImageBlock:
			// OpenAI assistant turns cannot carry images; drop them like
			// thinking so cross-provider replay degrades instead of failing.
		case *koine.ToolUseBlock:
			input := string(blk.Input)
			if input == "" {
				input = "{}"
			}
			p.ToolCalls = append(p.ToolCalls, openai.ChatCompletionMessageToolCallUnionParam{
				OfFunction: &openai.ChatCompletionMessageFunctionToolCallParam{
					ID: blk.ID,
					Function: openai.ChatCompletionMessageFunctionToolCallFunctionParam{
						Name:      blk.Name,
						Arguments: input,
					},
				},
			})
		default:
			return nil, fmt.Errorf("unsupported assistant block %q", b.BlockType())
		}
	}
	if text.Len() > 0 {
		p.Content = openai.ChatCompletionAssistantMessageParamContentUnion{OfString: openai.String(text.String())}
	}
	if text.Len() == 0 && len(p.ToolCalls) == 0 {
		return nil, nil
	}
	return []openai.ChatCompletionMessageParamUnion{{OfAssistant: &p}}, nil
}

func encodeUserMessage(m koine.Message) ([]openai.ChatCompletionMessageParamUnion, error) {
	if len(m.Blocks) == 1 {
		if t, ok := m.Blocks[0].(*koine.TextBlock); ok {
			return []openai.ChatCompletionMessageParamUnion{openai.UserMessage(t.Text)}, nil
		}
	}
	var parts []openai.ChatCompletionContentPartUnionParam
	for _, b := range m.Blocks {
		switch blk := b.(type) {
		case *koine.TextBlock:
			parts = append(parts, openai.TextContentPart(blk.Text))
		case *koine.ImageBlock:
			parts = append(parts, openai.ImageContentPart(openai.ChatCompletionContentPartImageImageURLParam{
				URL: imageURL(blk),
			}))
		default:
			return nil, fmt.Errorf("unsupported user block %q", b.BlockType())
		}
	}
	if len(parts) == 0 {
		return nil, nil
	}
	return []openai.ChatCompletionMessageParamUnion{openai.UserMessage(parts)}, nil
}

func toolResultText(tr *koine.ToolResultBlock) (string, error) {
	if len(tr.Result) > 0 {
		if len(tr.Content) > 0 {
			return "", fmt.Errorf("tool result %q sets both Content and Result", tr.ToolUseID)
		}
		return string(tr.Result), nil
	}
	var text strings.Builder
	for _, c := range tr.Content {
		t, ok := c.(*koine.TextBlock)
		if !ok {
			return "", fmt.Errorf("tool results support text only, got %q", c.BlockType())
		}
		text.WriteString(t.Text)
	}
	return text.String(), nil
}

func imageURL(b *koine.ImageBlock) string {
	if b.URL != "" {
		return b.URL
	}
	return "data:" + b.MIMEType + ";base64," + base64.StdEncoding.EncodeToString(b.Data)
}

func decodeResponse(acc *openai.ChatCompletionAccumulator, reasoning string, tools map[int64]*toolState, toolOrder []int64, finish string, usage koine.Usage) *koine.LanguageResponse {
	out := koine.Message{Role: koine.RoleAssistant}
	if reasoning != "" {
		out.Blocks = append(out.Blocks, &koine.ThinkingBlock{Text: reasoning})
	}
	if len(acc.Choices) > 0 {
		if content := acc.Choices[0].Message.Content; content != "" {
			out.Blocks = append(out.Blocks, &koine.TextBlock{Text: content})
		}
		if finish == "" {
			finish = acc.Choices[0].FinishReason
		}
	}
	for _, i := range toolOrder {
		st := tools[i]
		input := st.args.String()
		if input == "" {
			input = "{}"
		}
		out.Blocks = append(out.Blocks, &koine.ToolUseBlock{ID: st.id, Name: st.name, Input: json.RawMessage(input)})
	}
	return &koine.LanguageResponse{
		Message:    out,
		StopReason: mapFinishReason(finish),
		Usage:      usage,
		Model:      acc.Model,
		Provider:   Name,
	}
}

func mapFinishReason(finish string) koine.StopReason {
	switch finish {
	case "stop":
		return koine.StopEndTurn
	case "length":
		return koine.StopMaxTokens
	case "tool_calls", "function_call":
		return koine.StopToolUse
	case "content_filter":
		return koine.StopContentFilter
	default:
		return koine.StopOther
	}
}

func decodeUsage(u openai.CompletionUsage) (koine.Usage, bool) {
	if u.TotalTokens == 0 && u.PromptTokens == 0 && u.CompletionTokens == 0 {
		return koine.Usage{}, false
	}
	return koine.Usage{
		InputTokens:     int(u.PromptTokens),
		OutputTokens:    int(u.CompletionTokens),
		CacheReadTokens: int(u.PromptTokensDetails.CachedTokens),
		ReasoningTokens: int(u.CompletionTokensDetails.ReasoningTokens),
	}, true
}
