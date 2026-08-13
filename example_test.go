package koine_test

import (
	"context"
	"encoding/json"
	"fmt"

	"github.com/lf4096/koine"
	"github.com/lf4096/koine/provider/anthropic"
)

// Compile-only examples (no Output comment): they document usage in godoc
// without hitting the network.

func ExampleLanguageModel_Complete() {
	m := anthropic.New().LanguageModel("claude-sonnet-4-5")

	resp, err := m.Complete(context.Background(), &koine.LanguageRequest{
		Messages: []koine.Message{koine.UserText("Say hello in Greek.")},
	})
	if err != nil {
		panic(err)
	}
	fmt.Println(resp.Message.Text())
}

func ExampleLanguageModel_Stream() {
	ctx := context.Background()
	m := anthropic.New().LanguageModel("claude-sonnet-4-5")

	stream, err := m.Stream(ctx, &koine.LanguageRequest{
		Thinking: &koine.Thinking{Effort: koine.ThinkingMedium},
		Messages: []koine.Message{koine.UserText("Explain koine Greek briefly.")},
	})
	if err != nil {
		panic(err)
	}
	defer stream.Close()
	for ev := range stream.Events() {
		if ev.Type == koine.EventTextDelta {
			fmt.Print(ev.Text)
		}
	}
	if err := stream.Err(); err != nil {
		panic(err)
	}
}

func ExampleLanguageModel_Complete_toolLoop() {
	ctx := context.Background()
	m := anthropic.New().LanguageModel("claude-sonnet-4-5")

	req := &koine.LanguageRequest{
		Tools: []koine.Tool{{
			Name:        "get_weather",
			Description: "Get current weather for a city",
			InputSchema: map[string]any{
				"type":       "object",
				"properties": map[string]any{"city": map[string]any{"type": "string"}},
				"required":   []string{"city"},
			},
		}},
		Messages: []koine.Message{koine.UserText("Weather in Paris?")},
	}
	for {
		resp, err := m.Complete(ctx, req)
		if err != nil {
			panic(err)
		}
		if resp.StopReason != koine.StopToolUse {
			fmt.Println(resp.Message.Text())
			return
		}
		req.Messages = append(req.Messages, resp.Message)
		for _, call := range resp.Message.ToolUses() {
			req.Messages = append(req.Messages, koine.ToolResultJSON(call.ID, json.RawMessage(`{"temp":"22C"}`), false))
		}
	}
}
