package kieai_test

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/artschekoff/kie-ai-mcp/src/internal/kieai"
)

func TestPollUntilDone_Success(t *testing.T) {
	call := 0
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		call++
		state := "processing"
		if call >= 2 {
			state = "success"
		}
		resultJSON, _ := json.Marshal(map[string]any{
			"resultUrls": []string{"https://cdn.kie.ai/img.png"},
		})
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{
			"code": 200,
			"data": map[string]any{"state": state, "resultJson": string(resultJSON)},
		})
	}))
	defer srv.Close()

	cfg := kieai.Config{
		APIKey: "test-key", BaseURL: srv.URL, GetTaskPath: "/jobs/recordInfo",
		HTTPRetries: 1, TimeoutSeconds: 5, PollIntervalSec: 0.01, PollTimeoutSec: 5,
	}

	urls, err := kieai.PollUntilDone(cfg, "task-123")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(urls) == 0 {
		t.Fatal("expected image URLs, got none")
	}
	if urls[0] != "https://cdn.kie.ai/img.png" {
		t.Errorf("unexpected URL: %s", urls[0])
	}
	if call < 2 {
		t.Errorf("expected at least 2 poll calls, got %d", call)
	}
}

func TestPollUntilDone_Failure(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{
			"code": 200,
			"data": map[string]any{"state": "fail", "failMsg": "generation failed internally"},
		})
	}))
	defer srv.Close()

	cfg := kieai.Config{
		APIKey: "test-key", BaseURL: srv.URL, GetTaskPath: "/jobs/recordInfo",
		HTTPRetries: 1, TimeoutSeconds: 5, PollIntervalSec: 0.01, PollTimeoutSec: 5,
	}

	_, err := kieai.PollUntilDone(cfg, "task-fail")
	if err == nil {
		t.Fatal("expected error for failed task, got nil")
	}
}

func TestPollUntilDone_Timeout(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{
			"code": 200,
			"data": map[string]any{"state": "processing"},
		})
	}))
	defer srv.Close()

	// PollTimeoutSec=0 means deadline is immediately past
	cfg := kieai.Config{
		APIKey: "test-key", BaseURL: srv.URL, GetTaskPath: "/jobs/recordInfo",
		HTTPRetries: 1, TimeoutSeconds: 5, PollIntervalSec: 0.01, PollTimeoutSec: 0,
	}

	_, err := kieai.PollUntilDone(cfg, "task-timeout")
	if err == nil {
		t.Fatal("expected timeout error, got nil")
	}
}
