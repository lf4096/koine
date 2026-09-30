package openai

import (
	"encoding/json"
	"fmt"
	"slices"
	"strings"

	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/option"
	"github.com/openai/openai-go/v3/shared"

	"github.com/lf4096/koine"
)

func encodeChatRequest(model string, req *koine.LanguageRequest) (openai.ChatCompletionNewParams, []option.RequestOption, error) {
	o, _ := req.ProviderOptions[Name].(LanguageOptions)
	params := openai.ChatCompletionNewParams{Model: shared.ChatModel(model)}
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
		effort, err := reasoningEffort(req.Thinking)
		if err != nil {
			return openai.ChatCompletionNewParams{}, nil, err
		}
		params.ReasoningEffort = effortLevels[effort]
	}
	if rf := req.ResponseFormat; rf != nil {
		params.ResponseFormat = encodeChatResponseFormat(rf)
	}
	if !o.NoStreamUsage {
		params.StreamOptions = openai.ChatCompletionStreamOptionsParam{IncludeUsage: openai.Bool(true)}
	}
	if req.System != "" {
		params.Messages = append(params.Messages, openai.SystemMessage(req.System))
	}
	for _, m := range req.Messages {
		ms, err := encodeChatMessage(m, !o.NoReasoningReplay)
		if err != nil {
			return openai.ChatCompletionNewParams{}, nil, err
		}
		params.Messages = append(params.Messages, ms...)
	}
	for _, t := range req.Tools {
		params.Tools = append(params.Tools, encodeChatTool(t))
	}
	if req.ToolChoice != nil {
		params.ToolChoice = encodeChatToolChoice(req.ToolChoice)
	}
	return params, extraBody(o), nil
}

func encodeChatResponseFormat(rf *koine.ResponseFormat) openai.ChatCompletionNewParamsResponseFormatUnion {
	if len(rf.Schema) == 0 {
		return openai.ChatCompletionNewParamsResponseFormatUnion{
			OfJSONObject: &shared.ResponseFormatJSONObjectParam{},
		}
	}
	js := shared.ResponseFormatJSONSchemaJSONSchemaParam{Name: formatName(rf), Schema: rf.Schema}
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

func encodeChatTool(t koine.Tool) openai.ChatCompletionToolUnionParam {
	fn := shared.FunctionDefinitionParam{Name: t.Name}
	if t.Description != "" {
		fn.Description = openai.String(t.Description)
	}
	if len(t.InputSchema) > 0 {
		fn.Parameters = shared.FunctionParameters(t.InputSchema)
	}
	return openai.ChatCompletionFunctionTool(fn)
}

func encodeChatToolChoice(tc *koine.ToolChoice) openai.ChatCompletionToolChoiceOptionUnionParam {
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

func encodeChatMessage(m koine.Message, replayReasoning bool) ([]openai.ChatCompletionMessageParamUnion, error) {
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
		return encodeChatAssistant(m, replayReasoning)
	default:
		return encodeChatUser(m)
	}
}

func encodeChatAssistant(m koine.Message, replayReasoning bool) ([]openai.ChatCompletionMessageParamUnion, error) {
	var p openai.ChatCompletionAssistantMessageParam
	var text strings.Builder
	reasoning := map[string]any{}
	for _, b := range m.Blocks {
		switch blk := b.(type) {
		case *koine.TextBlock:
			text.WriteString(blk.Text)
		case *koine.ThinkingBlock:
			if !replayReasoning {
				continue
			}
			// Some vendors reject a tool-call turn that comes back without the
			// reasoning it streamed.
			if name, value, ok := reasoningFromRaw(blk.Raw); ok {
				prev, _ := reasoning[name].(string)
				reasoning[name] = prev + value
			}
		case *koine.ImageBlock:
			// OpenAI assistant turns cannot carry images; drop them like foreign
			// thinking so cross-provider replay degrades instead of failing.
		case *koine.ToolUseBlock:
			p.ToolCalls = append(p.ToolCalls, openai.ChatCompletionMessageToolCallUnionParam{
				OfFunction: &openai.ChatCompletionMessageFunctionToolCallParam{
					ID: blk.ID,
					Function: openai.ChatCompletionMessageFunctionToolCallFunctionParam{
						Name:      blk.Name,
						Arguments: callArguments(string(blk.Input)),
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
	if len(reasoning) > 0 {
		p.SetExtraFields(reasoning)
	}
	if text.Len() == 0 && len(p.ToolCalls) == 0 {
		return nil, nil
	}
	return []openai.ChatCompletionMessageParamUnion{{OfAssistant: &p}}, nil
}

func encodeChatUser(m koine.Message) ([]openai.ChatCompletionMessageParamUnion, error) {
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

// Raw under this provider can also be a Responses item; only a reasoning
// field replays on Chat Completions.
func reasoningFromRaw(raw *koine.ProviderRaw) (name, text string, ok bool) {
	if raw == nil || raw.Provider != Name {
		return "", "", false
	}
	var fields map[string]json.RawMessage
	if json.Unmarshal(raw.JSON, &fields) != nil {
		return "", "", false
	}
	for key, value := range fields {
		if slices.Contains(reasoningFields, key) && json.Unmarshal(value, &text) == nil {
			return key, text, true
		}
	}
	return "", "", false
}

func decodeChatResponse(acc *openai.ChatCompletionAccumulator, field, reasoning string, tools map[int64]*toolState, toolOrder []int64, finish string, usage koine.Usage) *koine.LanguageResponse {
	out := koine.Message{Role: koine.RoleAssistant}
	if reasoning != "" {
		raw, _ := json.Marshal(map[string]string{field: reasoning})
		out.Blocks = append(out.Blocks, &koine.ThinkingBlock{Text: reasoning, Raw: &koine.ProviderRaw{Provider: Name, JSON: raw}})
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
		out.Blocks = append(out.Blocks, &koine.ToolUseBlock{ID: st.id, Name: st.name, Input: json.RawMessage(callArguments(st.args.String()))})
	}
	return &koine.LanguageResponse{
		Message:    out,
		StopReason: chatStopReason(finish),
		Usage:      usage,
		Model:      acc.Model,
		Provider:   Name,
	}
}

func chatStopReason(finish string) koine.StopReason {
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

func decodeChatUsage(u openai.CompletionUsage) (koine.Usage, bool) {
	if u.TotalTokens == 0 && u.PromptTokens == 0 && u.CompletionTokens == 0 {
		return koine.Usage{}, false
	}
	d := u.PromptTokensDetails
	return koine.Usage{
		InputTokens:      int(max(u.PromptTokens-d.CachedTokens-d.CacheWriteTokens, 0)),
		OutputTokens:     int(u.CompletionTokens),
		CacheReadTokens:  int(d.CachedTokens),
		CacheWriteTokens: int(d.CacheWriteTokens),
		ReasoningTokens:  int(u.CompletionTokensDetails.ReasoningTokens),
	}, true
}
