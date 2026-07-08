package koine_test

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"reflect"
	"strings"
	"testing"

	"github.com/lf4096/koine"
)

func TestMessageJSONRoundTrip(t *testing.T) {
	msgs := []koine.Message{
		{Role: koine.RoleUser, Blocks: koine.Blocks{
			&koine.TextBlock{Text: "hello"},
			&koine.ImageBlock{MIMEType: "image/png", Data: []byte{1, 2, 3}},
		}},
		{Role: koine.RoleAssistant, Blocks: koine.Blocks{
			&koine.ThinkingBlock{
				Text:      "reasoning",
				Signature: "sig",
				Raw:       &koine.ProviderRaw{Provider: "anthropic", JSON: json.RawMessage(`{"type":"thinking","thinking":"reasoning","signature":"sig"}`)},
			},
			&koine.TextBlock{Text: "answer"},
			&koine.ToolUseBlock{ID: "t1", Name: "search", Input: json.RawMessage(`{"q":"x"}`)},
		}},
		{Role: koine.RoleTool, Blocks: koine.Blocks{
			&koine.ToolResultBlock{
				ToolUseID: "t1",
				Content:   koine.Blocks{&koine.TextBlock{Text: "result"}},
				IsError:   true,
			},
			&koine.ToolResultBlock{
				ToolUseID: "t2",
				Result:    json.RawMessage(`{"temp":22}`),
			},
		}},
	}
	for _, msg := range msgs {
		data, err := json.Marshal(msg)
		if err != nil {
			t.Fatalf("marshal: %v", err)
		}
		var back koine.Message
		if err := json.Unmarshal(data, &back); err != nil {
			t.Fatalf("unmarshal %s: %v", data, err)
		}
		if !reflect.DeepEqual(msg, back) {
			t.Errorf("round trip mismatch:\n in: %#v\nout: %#v", msg, back)
		}
	}
}

func TestBlocksJSONHasTypeDiscriminator(t *testing.T) {
	data, err := json.Marshal(koine.Blocks{&koine.TextBlock{Text: "hi"}})
	if err != nil {
		t.Fatal(err)
	}
	want := `[{"type":"text","text":"hi"}]`
	if string(data) != want {
		t.Errorf("got %s, want %s", data, want)
	}
}

func TestBlocksUnmarshalUnknownType(t *testing.T) {
	var blocks koine.Blocks
	err := json.Unmarshal([]byte(`[{"type":"bogus"}]`), &blocks)
	if err == nil || !strings.Contains(err.Error(), "bogus") {
		t.Errorf("want unknown block type error, got %v", err)
	}
}

func TestMessageHelpers(t *testing.T) {
	m := koine.Message{Role: koine.RoleAssistant, Blocks: koine.Blocks{
		&koine.TextBlock{Text: "a"},
		&koine.ThinkingBlock{Text: "skip"},
		&koine.TextBlock{Text: "b"},
		&koine.ToolUseBlock{ID: "t1", Name: "f"},
	}}
	if got := m.Text(); got != "ab" {
		t.Errorf("Text() = %q, want ab", got)
	}
	if uses := m.ToolUses(); len(uses) != 1 || uses[0].ID != "t1" {
		t.Errorf("ToolUses() = %#v", uses)
	}
	u := koine.UserText("hi")
	if u.Role != koine.RoleUser || u.Text() != "hi" {
		t.Errorf("UserText: %#v", u)
	}
	tr := koine.ToolResultText("t1", "out", false)
	if tr.Role != koine.RoleTool {
		t.Errorf("ToolResultText role = %q", tr.Role)
	}
}

func eventStream(events []koine.Event, resp *koine.LanguageResponse) *koine.LanguageStream {
	i := 0
	next := func() (koine.Event, error) {
		if i < len(events) {
			e := events[i]
			i++
			return e, nil
		}
		return koine.Event{}, io.EOF
	}
	final := func() (*koine.LanguageResponse, error) { return resp, nil }
	return koine.NewLanguageStream(next, final, nil)
}

func TestStreamScanner(t *testing.T) {
	events := []koine.Event{
		{Type: koine.EventTextDelta, Text: "a"},
		{Type: koine.EventStop, StopReason: koine.StopEndTurn},
	}
	s := eventStream(events, &koine.LanguageResponse{Model: "m"})
	var got []koine.Event
	for s.Next() {
		got = append(got, s.Event())
	}
	if err := s.Err(); err != nil {
		t.Fatalf("Err() = %v", err)
	}
	if len(got) != 2 || got[0].Text != "a" || got[1].StopReason != koine.StopEndTurn {
		t.Errorf("events = %#v", got)
	}
	if s.Response() == nil || s.Response().Model != "m" {
		t.Errorf("Response() = %#v", s.Response())
	}
	if s.Next() {
		t.Error("Next() after end must stay false")
	}
}

func TestStreamEventsIterator(t *testing.T) {
	events := []koine.Event{
		{Type: koine.EventTextDelta, Text: "a"},
		{Type: koine.EventTextDelta, Text: "b"},
		{Type: koine.EventStop, StopReason: koine.StopEndTurn},
	}
	s := eventStream(events, &koine.LanguageResponse{Model: "m"})
	var got []koine.Event
	for ev := range s.Events() {
		got = append(got, ev)
	}
	if err := s.Err(); err != nil {
		t.Fatalf("Err() = %v", err)
	}
	if len(got) != 3 || got[0].Text != "a" || got[1].Text != "b" {
		t.Errorf("events = %#v", got)
	}
	if s.Response() == nil || s.Response().Model != "m" {
		t.Errorf("Response() = %#v", s.Response())
	}

	s = eventStream(events, &koine.LanguageResponse{Model: "m"})
	n := 0
	for range s.Events() {
		n++
		break
	}
	if n != 1 {
		t.Errorf("break consumed %d events, want 1", n)
	}
	if !s.Next() {
		t.Error("Next() after break must resume the stream")
	}
}

func TestStreamError(t *testing.T) {
	fail := errors.New("boom")
	s := koine.NewLanguageStream(
		func() (koine.Event, error) { return koine.Event{}, fail },
		func() (*koine.LanguageResponse, error) { return nil, nil },
		nil,
	)
	if s.Next() {
		t.Fatal("Next() must be false on error")
	}
	if !errors.Is(s.Err(), fail) {
		t.Errorf("Err() = %v", s.Err())
	}
	if s.Response() != nil {
		t.Error("Response() must be nil on error")
	}
}

func TestStreamClose(t *testing.T) {
	closed := 0
	s := koine.NewLanguageStream(
		func() (koine.Event, error) { return koine.Event{Type: koine.EventTextDelta}, nil },
		func() (*koine.LanguageResponse, error) { return nil, nil },
		func() error { closed++; return nil },
	)
	if !s.Next() {
		t.Fatal("first Next() must succeed")
	}
	if err := s.Close(); err != nil {
		t.Fatal(err)
	}
	if err := s.Close(); err != nil {
		t.Fatal(err)
	}
	if closed != 1 {
		t.Errorf("close callback ran %d times, want 1", closed)
	}
	if s.Next() {
		t.Error("Next() after Close must be false")
	}
}

type fakeModel struct {
	events []koine.Event
	resp   *koine.LanguageResponse
}

func (f *fakeModel) Name() string { return "fake" }
func (f *fakeModel) Capabilities() koine.LanguageCapabilities {
	return koine.LanguageCapabilities{}
}
func (f *fakeModel) Complete(ctx context.Context, req *koine.LanguageRequest) (*koine.LanguageResponse, error) {
	s, err := f.Stream(ctx, req)
	if err != nil {
		return nil, err
	}
	return s.Collect()
}
func (f *fakeModel) Stream(ctx context.Context, req *koine.LanguageRequest) (*koine.LanguageStream, error) {
	return eventStream(f.events, f.resp), nil
}

func TestComplete(t *testing.T) {
	d := &fakeModel{
		events: []koine.Event{{Type: koine.EventTextDelta, Text: "x"}},
		resp:   &koine.LanguageResponse{StopReason: koine.StopEndTurn},
	}
	resp, err := d.Complete(context.Background(), &koine.LanguageRequest{Model: "m"})
	if err != nil {
		t.Fatal(err)
	}
	if resp.StopReason != koine.StopEndTurn {
		t.Errorf("resp = %#v", resp)
	}
}
