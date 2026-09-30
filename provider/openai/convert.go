package openai

import (
	"encoding/base64"
	"encoding/json"
	"fmt"
	"maps"
	"slices"
	"strings"

	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/option"
	"github.com/openai/openai-go/v3/packages/param"
	"github.com/openai/openai-go/v3/responses"
	"github.com/openai/openai-go/v3/shared"

	"github.com/lf4096/koine"
)

const reasoningSeparator = "\n\n"

var effortLevels = map[koine.ThinkingEffort]shared.ReasoningEffort{
	koine.ThinkingNone:    "none",
	koine.ThinkingMinimal: "minimal",
	koine.ThinkingLow:     "low",
	koine.ThinkingMedium:  "medium",
	koine.ThinkingHigh:    "high",
	koine.ThinkingXHigh:   "xhigh",
	koine.ThinkingMax:     "max",
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
	case budget <= 24576:
		return koine.ThinkingXHigh
	default:
		return koine.ThinkingMax
	}
}

func reasoningEffort(t *koine.Thinking) (koine.ThinkingEffort, error) {
	if t.Effort == koine.ThinkingNone && t.BudgetTokens > 0 {
		return "", fmt.Errorf("ThinkingNone cannot be combined with BudgetTokens")
	}
	if t.Effort == "" {
		return effortForBudget(t.BudgetTokens), nil
	}
	return t.Effort, nil
}

func extraBody(o LanguageOptions) []option.RequestOption {
	var reqOpts []option.RequestOption
	for k, v := range o.ExtraBody {
		reqOpts = append(reqOpts, option.WithJSONSet(k, v))
	}
	return reqOpts
}

func formatName(rf *koine.ResponseFormat) string {
	if rf.Name == "" {
		return "response"
	}
	return rf.Name
}

func callArguments(args string) string {
	if args == "" {
		return "{}"
	}
	return args
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

func encodeResponsesRequest(model string, req *koine.LanguageRequest) (responses.ResponseNewParams, []option.RequestOption, error) {
	if len(req.StopSequences) > 0 {
		return responses.ResponseNewParams{}, nil, fmt.Errorf("StopSequences are not supported by the Responses API")
	}
	o, _ := req.ProviderOptions[Name].(LanguageOptions)
	// Every call resends the whole conversation, so a reasoning item can
	// replay only from its encrypted content.
	params := responses.ResponseNewParams{
		Model:   shared.ResponsesModel(model),
		Store:   openai.Bool(false),
		Include: []responses.ResponseIncludable{responses.ResponseIncludableReasoningEncryptedContent},
	}
	if req.System != "" {
		params.Instructions = openai.String(req.System)
	}
	if req.MaxTokens > 0 {
		params.MaxOutputTokens = openai.Int(int64(req.MaxTokens))
	}
	if req.Temperature != nil {
		params.Temperature = openai.Float(*req.Temperature)
	}
	if req.TopP != nil {
		params.TopP = openai.Float(*req.TopP)
	}
	if req.Thinking != nil {
		effort, err := reasoningEffort(req.Thinking)
		if err != nil {
			return responses.ResponseNewParams{}, nil, err
		}
		params.Reasoning.Effort = effortLevels[effort]
		// OpenAI returns reasoning text only as summaries, and only on request.
		if effort != "" && effort != koine.ThinkingNone {
			params.Reasoning.Summary = shared.ReasoningSummaryAuto
		}
	}
	if rf := req.ResponseFormat; rf != nil {
		params.Text.Format = encodeResponsesFormat(rf)
	}
	input, err := encodeResponsesInput(req.Messages)
	if err != nil {
		return responses.ResponseNewParams{}, nil, err
	}
	params.Input = responses.ResponseNewParamsInputUnion{OfInputItemList: input}
	for _, t := range req.Tools {
		params.Tools = append(params.Tools, encodeResponsesTool(t))
	}
	if req.ToolChoice != nil {
		params.ToolChoice = encodeResponsesToolChoice(req.ToolChoice)
	}
	return params, extraBody(o), nil
}

func encodeResponsesFormat(rf *koine.ResponseFormat) responses.ResponseFormatTextConfigUnionParam {
	if len(rf.Schema) == 0 {
		return responses.ResponseFormatTextConfigUnionParam{OfJSONObject: &shared.ResponseFormatJSONObjectParam{}}
	}
	js := &responses.ResponseFormatTextJSONSchemaConfigParam{Name: formatName(rf), Schema: rf.Schema}
	if rf.Description != "" {
		js.Description = openai.String(rf.Description)
	}
	if rf.Strict {
		js.Strict = openai.Bool(true)
	}
	return responses.ResponseFormatTextConfigUnionParam{OfJSONSchema: js}
}

func encodeResponsesTool(t koine.Tool) responses.ToolUnionParam {
	// Responses validates function schemas strictly by default, which rejects
	// most JSON Schema outside its strict subset.
	fn := &responses.FunctionToolParam{Name: t.Name, Parameters: t.InputSchema, Strict: openai.Bool(false)}
	// Compatible vendors reject a Responses function tool without parameters.
	if len(fn.Parameters) == 0 {
		fn.Parameters = map[string]any{"type": "object", "properties": map[string]any{}}
	}
	if t.Description != "" {
		fn.Description = openai.String(t.Description)
	}
	return responses.ToolUnionParam{OfFunction: fn}
}

func encodeResponsesToolChoice(tc *koine.ToolChoice) responses.ResponseNewParamsToolChoiceUnion {
	switch tc.Mode {
	case koine.ToolChoiceNone:
		return responses.ResponseNewParamsToolChoiceUnion{OfToolChoiceMode: param.NewOpt(responses.ToolChoiceOptionsNone)}
	case koine.ToolChoiceRequired:
		return responses.ResponseNewParamsToolChoiceUnion{OfToolChoiceMode: param.NewOpt(responses.ToolChoiceOptionsRequired)}
	case koine.ToolChoiceTool:
		return responses.ResponseNewParamsToolChoiceUnion{OfFunctionTool: &responses.ToolChoiceFunctionParam{Name: tc.Name}}
	default:
		return responses.ResponseNewParamsToolChoiceUnion{OfToolChoiceMode: param.NewOpt(responses.ToolChoiceOptionsAuto)}
	}
}

func encodeResponsesInput(msgs []koine.Message) (responses.ResponseInputParam, error) {
	var input responses.ResponseInputParam
	for _, m := range msgs {
		switch m.Role {
		case koine.RoleTool:
			for _, b := range m.Blocks {
				tr, ok := b.(*koine.ToolResultBlock)
				if !ok {
					return nil, fmt.Errorf("tool message requires tool_result blocks, got %q", b.BlockType())
				}
				item, err := encodeResponsesToolOutput(tr)
				if err != nil {
					return nil, err
				}
				input = append(input, item)
			}
		case koine.RoleAssistant:
			items, err := encodeResponsesAssistant(m.Blocks)
			if err != nil {
				return nil, err
			}
			input = append(input, items...)
		default:
			item, ok, err := encodeResponsesUser(m.Blocks)
			if err != nil {
				return nil, err
			}
			if ok {
				input = append(input, item)
			}
		}
	}
	return input, nil
}

func encodeResponsesToolOutput(tr *koine.ToolResultBlock) (responses.ResponseInputItemUnionParam, error) {
	var item responses.ResponseInputItemUnionParam
	if len(tr.Result) > 0 || !slices.ContainsFunc(tr.Content, func(b koine.Block) bool { return b.BlockType() == koine.BlockImage }) {
		text, err := toolResultText(tr)
		if err != nil {
			return item, err
		}
		item = responses.ResponseInputItemParamOfFunctionCallOutput(text)
	} else {
		var output responses.ResponseFunctionCallOutputItemListParam
		for _, c := range tr.Content {
			switch blk := c.(type) {
			case *koine.TextBlock:
				output = append(output, responses.ResponseFunctionCallOutputItemUnionParam{OfInputText: &responses.ResponseInputTextContentParam{Text: blk.Text}})
			case *koine.ImageBlock:
				output = append(output, responses.ResponseFunctionCallOutputItemUnionParam{OfInputImage: &responses.ResponseInputImageContentParam{ImageURL: openai.String(imageURL(blk))}})
			default:
				return item, fmt.Errorf("unsupported tool result content %q", c.BlockType())
			}
		}
		item = responses.ResponseInputItemParamOfFunctionCallOutput(output)
	}
	item.OfFunctionCallOutput.CallID = openai.String(tr.ToolUseID)
	return item, nil
}

func encodeResponsesAssistant(blocks koine.Blocks) ([]responses.ResponseInputItemUnionParam, error) {
	var items []responses.ResponseInputItemUnionParam
	for _, b := range blocks {
		switch blk := b.(type) {
		case *koine.ThinkingBlock:
			if item, ok := itemFromRaw(blk.Raw, "reasoning"); ok {
				items = append(items, item)
			}
		case *koine.TextBlock:
			if item, ok := itemFromRaw(blk.Raw, "message"); ok {
				items = append(items, item)
				continue
			}
			items = append(items, responses.ResponseInputItemParamOfMessage(blk.Text, responses.EasyInputMessageRoleAssistant))
		case *koine.ToolUseBlock:
			if item, ok := itemFromRaw(blk.Raw, "function_call"); ok {
				items = append(items, item)
				continue
			}
			items = append(items, responses.ResponseInputItemParamOfFunctionCall(callArguments(string(blk.Input)), blk.ID, blk.Name))
		case *koine.ImageBlock:
		default:
			return nil, fmt.Errorf("unsupported assistant block %q", b.BlockType())
		}
	}
	return items, nil
}

func encodeResponsesUser(blocks koine.Blocks) (responses.ResponseInputItemUnionParam, bool, error) {
	if len(blocks) == 1 {
		if t, ok := blocks[0].(*koine.TextBlock); ok {
			return responses.ResponseInputItemParamOfMessage(t.Text, responses.EasyInputMessageRoleUser), true, nil
		}
	}
	var parts responses.ResponseInputMessageContentListParam
	for _, b := range blocks {
		switch blk := b.(type) {
		case *koine.TextBlock:
			parts = append(parts, responses.ResponseInputContentParamOfInputText(blk.Text))
		case *koine.ImageBlock:
			parts = append(parts, responses.ResponseInputContentUnionParam{OfInputImage: &responses.ResponseInputImageParam{
				ImageURL: openai.String(imageURL(blk)),
				Detail:   responses.ResponseInputImageDetailAuto,
			}})
		default:
			return responses.ResponseInputItemUnionParam{}, false, fmt.Errorf("unsupported user block %q", b.BlockType())
		}
	}
	if len(parts) == 0 {
		return responses.ResponseInputItemUnionParam{}, false, nil
	}
	return responses.ResponseInputItemParamOfMessage(parts, responses.EasyInputMessageRoleUser), true, nil
}

func itemFromRaw(raw *koine.ProviderRaw, itemType string) (responses.ResponseInputItemUnionParam, bool) {
	if raw == nil || raw.Provider != Name {
		return responses.ResponseInputItemUnionParam{}, false
	}
	var head struct {
		Type string `json:"type"`
		Role string `json:"role"`
	}
	if json.Unmarshal(raw.JSON, &head) != nil || head.Type != itemType {
		return responses.ResponseInputItemUnionParam{}, false
	}
	// A message item carries its own role; replay must never turn it into a
	// system or developer instruction.
	if itemType == "message" && head.Role != "assistant" {
		return responses.ResponseInputItemUnionParam{}, false
	}
	return param.Override[responses.ResponseInputItemUnionParam](raw.JSON), true
}

func callInput(item responses.ResponseOutputItemUnion) json.RawMessage {
	return json.RawMessage(callArguments(item.Arguments.OfString))
}

func decodeResponsesResult(r *responses.Response, items map[int64]responses.ResponseOutputItemUnion) *koine.LanguageResponse {
	out := koine.Message{Role: koine.RoleAssistant}
	for _, i := range slices.Sorted(maps.Keys(items)) {
		item := items[i]
		raw := &koine.ProviderRaw{Provider: Name, JSON: json.RawMessage(item.RawJSON())}
		switch item.Type {
		case "reasoning":
			out.Blocks = append(out.Blocks, &koine.ThinkingBlock{Text: reasoningText(item), Raw: raw})
		case "message":
			var text strings.Builder
			for _, c := range item.Content {
				text.WriteString(c.Text)
				text.WriteString(c.Refusal)
			}
			out.Blocks = append(out.Blocks, &koine.TextBlock{Text: text.String(), Raw: raw})
		case "function_call":
			out.Blocks = append(out.Blocks, &koine.ToolUseBlock{ID: item.CallID, Name: item.Name, Input: callInput(item), Raw: raw})
		}
	}
	usage, _ := decodeResponsesUsage(r.Usage)
	return &koine.LanguageResponse{
		Message:    out,
		StopReason: responsesStopReason(r, items),
		Usage:      usage,
		Model:      string(r.Model),
		Provider:   Name,
	}
}

// Summaries are all OpenAI returns; other vendors return the full reasoning
// as content instead.
func reasoningText(item responses.ResponseOutputItemUnion) string {
	var parts []string
	for _, s := range item.Summary {
		parts = append(parts, s.Text)
	}
	if len(parts) == 0 {
		for _, c := range item.Content {
			if c.Type == "reasoning_text" {
				parts = append(parts, c.Text)
			}
		}
	}
	return strings.Join(parts, reasoningSeparator)
}

func responsesStopReason(r *responses.Response, items map[int64]responses.ResponseOutputItemUnion) koine.StopReason {
	if r.Status == responses.ResponseStatusIncomplete {
		switch r.IncompleteDetails.Reason {
		case "max_output_tokens":
			return koine.StopMaxTokens
		case "content_filter":
			return koine.StopContentFilter
		}
		return koine.StopOther
	}
	stop := koine.StopEndTurn
	for _, item := range items {
		if item.Type == "function_call" {
			return koine.StopToolUse
		}
		for _, c := range item.Content {
			if c.Type == "refusal" {
				stop = koine.StopContentFilter
			}
		}
	}
	return stop
}

func decodeResponsesUsage(u responses.ResponseUsage) (koine.Usage, bool) {
	if u.TotalTokens == 0 && u.InputTokens == 0 && u.OutputTokens == 0 {
		return koine.Usage{}, false
	}
	d := u.InputTokensDetails
	return koine.Usage{
		InputTokens:      int(max(u.InputTokens-d.CachedTokens-d.CacheWriteTokens, 0)),
		OutputTokens:     int(u.OutputTokens),
		CacheReadTokens:  int(d.CachedTokens),
		CacheWriteTokens: int(d.CacheWriteTokens),
		ReasoningTokens:  int(u.OutputTokensDetails.ReasoningTokens),
	}, true
}
