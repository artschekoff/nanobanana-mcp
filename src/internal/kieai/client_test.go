package kieai_test

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/artschekoff/kie-ai-mcp/internal/kieai"
)

func TestCreateTask_Success(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			t.Errorf("expected POST, got %s", r.Method)
		}
		if r.Header.Get("Authorization") != "Bearer test-key" {
			t.Errorf("missing auth header")
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{
			"code": 200,
			"data": map[string]any{"taskId": "task-123"},
		})
	}))
	defer srv.Close()

	cfg := kieai.Config{
		APIKey: "test-key", BaseURL: srv.URL,
		HTTPRetries: 1, TimeoutSeconds: 5, CreateTaskPath: "/jobs/createTask",
	}
	req := kieai.CreateTaskRequest{
		Model: "nano-banana-pro",
		Input: kieai.TaskInput{Prompt: "a cat", AspectRatio: "1:1", Resolution: "1K", OutputFormat: "png"},
	}

	taskID, err := kieai.CreateTask(cfg, req)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if taskID != "task-123" {
		t.Errorf("expected task-123, got %q", taskID)
	}
}

func TestCreateTask_APIError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{"code": 400, "msg": "bad request"})
	}))
	defer srv.Close()

	cfg := kieai.Config{
		APIKey: "test-key", BaseURL: srv.URL,
		HTTPRetries: 1, TimeoutSeconds: 5, CreateTaskPath: "/jobs/createTask",
	}
	req := kieai.CreateTaskRequest{
		Model: "nano-banana-pro",
		Input: kieai.TaskInput{Prompt: "a cat", AspectRatio: "1:1", Resolution: "1K", OutputFormat: "png"},
	}

	_, err := kieai.CreateTask(cfg, req)
	if err == nil {
		t.Fatal("expected error for API code 400, got nil")
	}
}

func TestCreateTask_RetryOnTransient(t *testing.T) {
	calls := 0
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls++
		if calls < 2 {
			w.WriteHeader(http.StatusServiceUnavailable)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{
			"code": 200,
			"data": map[string]any{"taskId": "task-retry"},
		})
	}))
	defer srv.Close()

	cfg := kieai.Config{
		APIKey: "test-key", BaseURL: srv.URL,
		HTTPRetries: 3, TimeoutSeconds: 5, CreateTaskPath: "/jobs/createTask",
		HTTPRetryBackoff: 0.01,
	}
	req := kieai.CreateTaskRequest{
		Model: "nano-banana-pro",
		Input: kieai.TaskInput{Prompt: "a cat", AspectRatio: "1:1", Resolution: "1K", OutputFormat: "png"},
	}

	taskID, err := kieai.CreateTask(cfg, req)
	if err != nil {
		t.Fatalf("expected retry to succeed: %v", err)
	}
	if taskID != "task-retry" {
		t.Errorf("expected task-retry, got %q", taskID)
	}
	if calls < 2 {
		t.Errorf("expected at least 2 calls (retry), got %d", calls)
	}
}

func TestGetTask_Success(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Query().Get("taskId") != "task-123" {
			t.Errorf("expected taskId=task-123, got %q", r.URL.Query().Get("taskId"))
		}
		resultJSON, _ := json.Marshal(map[string]any{"resultUrls": []string{"https://cdn.kie.ai/img.png"}})
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{
			"code": 200,
			"data": map[string]any{"state": "success", "resultJson": string(resultJSON)},
		})
	}))
	defer srv.Close()

	cfg := kieai.Config{
		APIKey: "test-key", BaseURL: srv.URL,
		HTTPRetries: 1, TimeoutSeconds: 5, GetTaskPath: "/jobs/recordInfo",
	}

	data, err := kieai.GetTask(cfg, "task-123")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if data.State != "success" {
		t.Errorf("expected state=success, got %q", data.State)
	}
}
