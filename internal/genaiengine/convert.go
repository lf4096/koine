package genaiengine

import (
	"encoding/base64"
	"encoding/json"
	"fmt"
	"strings"

	"google.golang.org/genai"

	"github.com/lf4096/koine"
)

var effortLevels = map[koine.ThinkingEffort]genai.ThinkingLevel{
	koine.ThinkingMinimal: genai.ThinkingLevelMinimal,
	koine.ThinkingLow:     genai.ThinkingLevelLow,
	koine.ThinkingMedium:  genai.ThinkingLevelMedium,
	koine.ThinkingHigh:    genai.ThinkingLevelHigh,
	koine.ThinkingXHigh:   genai.ThinkingLevelHigh,
	koine.ThinkingMax:     genai.ThinkingLevelHigh,
}

func (e *Engine) encodeRequest(req *koine.LanguageRequest) ([]*genai.Content, *genai.GenerateContentConfig, error) {
	config := &genai.GenerateContentConfig{StopSequences: req.StopSequences}
	if req.System != "" {
		config.SystemInstruction = &genai.Content{Parts: []*genai.Part{{Text: req.System}}}
	}
	if req.Temperature != nil {
		config.Temperature = new(float32(*req.Temperature))
	}
	if req.TopP != nil {
		config.TopP = new(float32(*req.TopP))
	}
	if req.MaxTokens > 0 {
		config.MaxOutputTokens = int32(req.MaxTokens)
	}
	if req.Thinking != nil {
		switch {
		case req.Thinking.Effort == koine.ThinkingNone:
			if req.Thinking.BudgetTokens > 0 {
				return nil, nil, fmt.Errorf("ThinkingNone cannot be combined with BudgetTokens")
			}
			config.ThinkingConfig = &genai.ThinkingConfig{ThinkingBudget: new(int32(0))}
		case req.Thinking.BudgetTokens > 0:
			config.ThinkingConfig = &genai.ThinkingConfig{IncludeThoughts: true, ThinkingBudget: new(int32(req.Thinking.BudgetTokens))}
		default:
			if level, ok := effortLevels[req.Thinking.Effort]; ok {
				config.ThinkingConfig = &genai.ThinkingConfig{IncludeThoughts: true, ThinkingLevel: level}
			}
		}
	}
	if rf := req.ResponseFormat; rf != nil {
		// The generateContent API rejects JSON response constraints combined
		// with function declarations.
		if len(req.Tools) > 0 {
			return nil, nil, fmt.Errorf("ResponseFormat cannot be combined with Tools")
		}
		config.ResponseMIMEType = "application/json"
		if len(rf.Schema) > 0 {
			config.ResponseJsonSchema = rf.Schema
		}
	}
	if len(req.Tools) > 0 {
		tool := &genai.Tool{}
		for _, t := range req.Tools {
			tool.FunctionDeclarations = append(tool.FunctionDeclarations, encodeTool(t))
		}
		config.Tools = []*genai.Tool{tool}
	}
	if req.ToolChoice != nil {
		config.ToolConfig = encodeToolChoice(req.ToolChoice)
	}
	if o, ok := req.ProviderOptions[e.Name].(LanguageOptions); ok {
		config.SafetySettings = o.SafetySettings
		if o.ThinkingConfig != nil {
			config.ThinkingConfig = o.ThinkingConfig
		}
	}
	contents, err := e.encodeMessages(req.Messages)
	if err != nil {
		return nil, nil, err
	}
	return contents, config, nil
}

func encodeTool(t koine.Tool) *genai.FunctionDeclaration {
	fd := &genai.FunctionDeclaration{Name: t.Name, Description: t.Description}
	if len(t.InputSchema) > 0 {
		fd.ParametersJsonSchema = t.InputSchema
	}
	return fd
}

func encodeToolChoice(tc *koine.ToolChoice) *genai.ToolConfig {
	fcc := &genai.FunctionCallingConfig{}
	switch tc.Mode {
	case koine.ToolChoiceNone:
		fcc.Mode = genai.FunctionCallingConfigModeNone
	case koine.ToolChoiceRequired:
		fcc.Mode = genai.FunctionCallingConfigModeAny
	case koine.ToolChoiceTool:
		fcc.Mode = genai.FunctionCallingConfigModeAny
		fcc.AllowedFunctionNames = []string{tc.Name}
	default:
		fcc.Mode = genai.FunctionCallingConfigModeAuto
	}
	return &genai.ToolConfig{FunctionCallingConfig: fcc}
}

// encodeMessages reshapes canonical messages to genai contents. It tracks
// tool_use id -> function name because FunctionResponse parts must repeat the
// function name, which ToolResultBlock does not carry.
func (e *Engine) encodeMessages(msgs []koine.Message) ([]*genai.Content, error) {
	toolNames := map[string]string{}
	var out []*genai.Content
	for _, m := range msgs {
		var (
			parts []*genai.Part
			role  genai.Role
			err   error
		)
		switch m.Role {
		case koine.RoleAssistant:
			role = genai.RoleModel
			parts, err = e.encodeAssistantParts(m.Blocks, toolNames)
		case koine.RoleTool:
			role = genai.RoleUser
			parts, err = e.encodeToolParts(m.Blocks, toolNames)
		default:
			role = genai.RoleUser
			parts, err = e.encodeUserParts(m.Blocks)
		}
		if err != nil {
			return nil, err
		}
		if len(parts) > 0 {
			out = append(out, &genai.Content{Role: string(role), Parts: parts})
		}
	}
	return out, nil
}

func (e *Engine) encodeAssistantParts(blocks koine.Blocks, toolNames map[string]string) ([]*genai.Part, error) {
	var parts []*genai.Part
	for _, b := range blocks {
		switch blk := b.(type) {
		case *koine.TextBlock:
			if p, ok := e.partFromRaw(blk.Raw); ok {
				parts = append(parts, p)
			} else {
				parts = append(parts, &genai.Part{Text: blk.Text})
			}
		case *koine.ThinkingBlock:
			if p, ok := e.partFromRaw(blk.Raw); ok {
				parts = append(parts, p)
			}
			// Foreign thinking cannot be replayed; drop it.
		case *koine.ToolUseBlock:
			toolNames[blk.ID] = blk.Name
			if p, ok := e.partFromRaw(blk.Raw); ok {
				parts = append(parts, p)
				continue
			}
			var args map[string]any
			if len(blk.Input) > 0 {
				if err := json.Unmarshal(blk.Input, &args); err != nil {
					return nil, fmt.Errorf("tool call %q input: %w", blk.Name, err)
				}
			}
			parts = append(parts, &genai.Part{FunctionCall: &genai.FunctionCall{
				ID:   wireID(blk.ID),
				Name: blk.Name,
				Args: args,
			}})
		case *koine.ImageBlock:
			parts = append(parts, imagePart(blk))
		default:
			return nil, fmt.Errorf("unsupported assistant block %q", b.BlockType())
		}
	}
	return parts, nil
}

func (e *Engine) encodeToolParts(blocks koine.Blocks, toolNames map[string]string) ([]*genai.Part, error) {
	var parts []*genai.Part
	for _, b := range blocks {
		tr, ok := b.(*koine.ToolResultBlock)
		if !ok {
			return nil, fmt.Errorf("tool message requires tool_result blocks, got %q", b.BlockType())
		}
		name := toolNames[tr.ToolUseID]
		if name == "" {
			return nil, fmt.Errorf("tool result %q has no matching tool call in the conversation", tr.ToolUseID)
		}
		response, err := e.toolResponse(tr)
		if err != nil {
			return nil, err
		}
		parts = append(parts, &genai.Part{FunctionResponse: &genai.FunctionResponse{
			ID:       wireID(tr.ToolUseID),
			Name:     name,
			Response: response,
		}})
	}
	return parts, nil
}

// toolResponse builds the FunctionResponse payload, which the API takes as a
// JSON object: a structured Result object passes through natively, anything
// else wraps under the conventional "output"/"error" key.
func (e *Engine) toolResponse(tr *koine.ToolResultBlock) (map[string]any, error) {
	if len(tr.Result) > 0 {
		if len(tr.Content) > 0 {
			return nil, fmt.Errorf("tool result %q sets both Content and Result", tr.ToolUseID)
		}
		var v any
		if err := json.Unmarshal(tr.Result, &v); err != nil {
			return nil, fmt.Errorf("tool result %q: %w", tr.ToolUseID, err)
		}
		if tr.IsError {
			return map[string]any{"error": v}, nil
		}
		if m, ok := v.(map[string]any); ok {
			return m, nil
		}
		return map[string]any{"output": v}, nil
	}
	var text strings.Builder
	for _, c := range tr.Content {
		t, ok := c.(*koine.TextBlock)
		if !ok {
			return nil, fmt.Errorf("tool results support text only, got %q", c.BlockType())
		}
		text.WriteString(t.Text)
	}
	if tr.IsError {
		return map[string]any{"error": text.String()}, nil
	}
	return map[string]any{"output": text.String()}, nil
}

func (e *Engine) encodeUserParts(blocks koine.Blocks) ([]*genai.Part, error) {
	var parts []*genai.Part
	for _, b := range blocks {
		switch blk := b.(type) {
		case *koine.TextBlock:
			parts = append(parts, &genai.Part{Text: blk.Text})
		case *koine.ImageBlock:
			parts = append(parts, imagePart(blk))
		default:
			return nil, fmt.Errorf("unsupported user block %q", b.BlockType())
		}
	}
	return parts, nil
}

func (e *Engine) partFromRaw(raw *koine.ProviderRaw) (*genai.Part, bool) {
	if !e.acceptsRaw(raw) {
		return nil, false
	}
	var p genai.Part
	if err := json.Unmarshal(raw.JSON, &p); err != nil {
		return nil, false
	}
	return &p, true
}

// wireID strips synthetic ids so the API never sees ids it did not mint.
func wireID(id string) string {
	if strings.HasPrefix(id, syntheticIDPrefix) {
		return ""
	}
	return id
}

func imagePart(b *koine.ImageBlock) *genai.Part {
	if b.URL != "" {
		return &genai.Part{FileData: &genai.FileData{FileURI: b.URL, MIMEType: b.MIMEType}}
	}
	return &genai.Part{InlineData: &genai.Blob{MIMEType: b.MIMEType, Data: b.Data}}
}

// accumulator folds streamed parts into canonical blocks. Consecutive thought
// or text parts merge into one block; their thought signatures are preserved
// by rebuilding the provider part as ProviderRaw at close.
type accumulator struct {
	name   string
	blocks koine.Blocks

	thinkingOpen bool
	thinkingText strings.Builder
	thinkingSig  []byte

	textBlock *koine.TextBlock
	textSig   []byte

	usage  koine.Usage
	finish genai.FinishReason
	model  string
	tools  int
}

func (a *accumulator) addResponse(resp *genai.GenerateContentResponse) {
	if resp.ModelVersion != "" {
		a.model = resp.ModelVersion
	}
	if u := resp.UsageMetadata; u != nil {
		a.usage = koine.Usage{
			InputTokens:     int(u.PromptTokenCount),
			OutputTokens:    int(u.CandidatesTokenCount + u.ThoughtsTokenCount),
			CacheReadTokens: int(u.CachedContentTokenCount),
			ReasoningTokens: int(u.ThoughtsTokenCount),
		}
	}
	if len(resp.Candidates) > 0 && resp.Candidates[0].FinishReason != "" {
		a.finish = resp.Candidates[0].FinishReason
	}
}

func (a *accumulator) addThinking(part *genai.Part) {
	a.closeText()
	a.thinkingOpen = true
	a.thinkingText.WriteString(part.Text)
	if len(part.ThoughtSignature) > 0 {
		a.thinkingSig = part.ThoughtSignature
	}
}

func (a *accumulator) addText(part *genai.Part) {
	a.closeThinking()
	if a.textBlock == nil {
		a.textBlock = &koine.TextBlock{}
		a.blocks = append(a.blocks, a.textBlock)
	}
	a.textBlock.Text += part.Text
	if len(part.ThoughtSignature) > 0 {
		a.textSig = part.ThoughtSignature
	}
}

func (a *accumulator) addFunctionCall(part *genai.Part, id string) json.RawMessage {
	a.closeThinking()
	a.closeText()
	a.tools++
	input := json.RawMessage("{}")
	if len(part.FunctionCall.Args) > 0 {
		if data, err := json.Marshal(part.FunctionCall.Args); err == nil {
			input = data
		}
	}
	a.blocks = append(a.blocks, &koine.ToolUseBlock{
		ID:    id,
		Name:  part.FunctionCall.Name,
		Input: input,
		Raw:   a.rawPart(part),
	})
	return input
}

func (a *accumulator) addOther(part *genai.Part) {
	if part.InlineData == nil {
		return
	}
	a.closeThinking()
	a.closeText()
	a.blocks = append(a.blocks, &koine.ImageBlock{
		MIMEType: part.InlineData.MIMEType,
		Data:     part.InlineData.Data,
	})
}

func (a *accumulator) closeThinking() {
	if !a.thinkingOpen {
		return
	}
	part := &genai.Part{Text: a.thinkingText.String(), Thought: true, ThoughtSignature: a.thinkingSig}
	a.blocks = append(a.blocks, &koine.ThinkingBlock{
		Text:      part.Text,
		Signature: base64.StdEncoding.EncodeToString(a.thinkingSig),
		Raw:       a.rawPart(part),
	})
	a.thinkingOpen = false
	a.thinkingText.Reset()
	a.thinkingSig = nil
}

func (a *accumulator) closeText() {
	if a.textBlock == nil {
		return
	}
	if len(a.textSig) > 0 {
		part := &genai.Part{Text: a.textBlock.Text, ThoughtSignature: a.textSig}
		a.textBlock.Raw = a.rawPart(part)
	}
	a.textBlock = nil
	a.textSig = nil
}

func (a *accumulator) stopReason() koine.StopReason {
	switch a.finish {
	case genai.FinishReasonStop:
		if a.tools > 0 {
			return koine.StopToolUse
		}
		return koine.StopEndTurn
	case genai.FinishReasonMaxTokens:
		return koine.StopMaxTokens
	case genai.FinishReasonSafety, genai.FinishReasonRecitation, genai.FinishReasonBlocklist,
		genai.FinishReasonProhibitedContent, genai.FinishReasonSPII, genai.FinishReasonImageSafety:
		return koine.StopContentFilter
	default:
		return koine.StopOther
	}
}

func (a *accumulator) response() *koine.LanguageResponse {
	a.closeThinking()
	a.closeText()
	return &koine.LanguageResponse{
		Message:    koine.Message{Role: koine.RoleAssistant, Blocks: a.blocks},
		StopReason: a.stopReason(),
		Usage:      a.usage,
		Model:      a.model,
		Provider:   a.name,
	}
}

func (a *accumulator) rawPart(part *genai.Part) *koine.ProviderRaw {
	data, err := json.Marshal(part)
	if err != nil {
		return nil
	}
	return &koine.ProviderRaw{Provider: a.name, JSON: data}
}
