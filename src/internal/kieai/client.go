package kieai

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"mime/multipart"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"time"
)

func buildURL(base, path string) string {
	if strings.HasPrefix(path, "http://") || strings.HasPrefix(path, "https://") {
		return path
	}
	return strings.TrimRight(base, "/") + "/" + strings.TrimLeft(path, "/")
}

func isTransient(code int) bool {
	switch code {
	case 408, 425, 429, 500, 502, 503, 504:
		return true
	}
	return false
}

func retryDelay(attempt int, backoff float64) time.Duration {
	if backoff < 0.01 {
		backoff = 0.01
	}
	return time.Duration(backoff * float64(attempt) * float64(time.Second))
}

func doRequest(method, url, apiKey string, body []byte, timeoutSec, retries int, backoff float64) ([]byte, error) {
	client := &http.Client{Timeout: time.Duration(timeoutSec) * time.Second}
	if retries < 1 {
		retries = 1
	}
	var lastErr error
	for attempt := 1; attempt <= retries; attempt++ {
		var bodyReader io.Reader
		if body != nil {
			bodyReader = bytes.NewReader(body)
		}
		req, err := http.NewRequest(method, url, bodyReader)
		if err != nil {
			return nil, err
		}
		req.Header.Set("Authorization", "Bearer "+apiKey)
		if body != nil {
			req.Header.Set("Content-Type", "application/json")
		}

		resp, err := client.Do(req)
		if err != nil {
			lastErr = fmt.Errorf("network error: %w", err)
			if attempt < retries {
				time.Sleep(retryDelay(attempt, backoff))
			}
			continue
		}
		defer resp.Body.Close()
		raw, readErr := io.ReadAll(resp.Body)
		if readErr != nil {
			lastErr = fmt.Errorf("read response body: %w", readErr)
			if attempt < retries {
				time.Sleep(retryDelay(attempt, backoff))
			}
			continue
		}
		if resp.StatusCode >= 300 {
			lastErr = fmt.Errorf("HTTP %d: %s", resp.StatusCode, string(raw))
			if attempt < retries && isTransient(resp.StatusCode) {
				time.Sleep(retryDelay(attempt, backoff))
				continue
			}
			return nil, lastErr
		}
		return raw, nil
	}
	return nil, lastErr
}

// CreateTask submits an image generation job to Kie AI and returns the task ID.
func CreateTask(cfg Config, req CreateTaskRequest) (string, error) {
	if cfg.APIKey == "" {
		return "", fmt.Errorf("KIE_AI_API_KEY is required")
	}
	body, err := json.Marshal(req)
	if err != nil {
		return "", err
	}
	url := buildURL(cfg.BaseURL, cfg.CreateTaskPath)
	raw, err := doRequest(http.MethodPost, url, cfg.APIKey, body, cfg.TimeoutSeconds, cfg.HTTPRetries, cfg.HTTPRetryBackoff)
	if err != nil {
		return "", err
	}
	var resp APIResponse
	if err := json.Unmarshal(raw, &resp); err != nil {
		return "", fmt.Errorf("parse response: %w", err)
	}
	if resp.Code != 200 {
		return "", fmt.Errorf("API error (code %d): %s", resp.Code, resp.Msg)
	}
	if len(resp.Data) == 0 {
		return "", fmt.Errorf("API returned empty data field")
	}
	var data CreateTaskData
	if err := json.Unmarshal(resp.Data, &data); err != nil {
		return "", fmt.Errorf("parse task data: %w", err)
	}
	if data.TaskID == "" {
		return "", fmt.Errorf("no task ID in response")
	}
	return data.TaskID, nil
}

// GetTask fetches the current status and result of a task.
func GetTask(cfg Config, taskID string) (*PollData, error) {
	if cfg.APIKey == "" {
		return nil, fmt.Errorf("KIE_AI_API_KEY is required")
	}
	url := buildURL(cfg.BaseURL, cfg.GetTaskPath) + "?taskId=" + taskID
	raw, err := doRequest(http.MethodGet, url, cfg.APIKey, nil, cfg.TimeoutSeconds, cfg.HTTPRetries, cfg.HTTPRetryBackoff)
	if err != nil {
		return nil, err
	}
	var resp APIResponse
	if err := json.Unmarshal(raw, &resp); err != nil {
		return nil, fmt.Errorf("parse response: %w", err)
	}
	if resp.Code != 200 {
		return nil, fmt.Errorf("API error (code %d): %s", resp.Code, resp.Msg)
	}
	if len(resp.Data) == 0 {
		return nil, fmt.Errorf("API returned empty data field")
	}
	var data PollData
	if err := json.Unmarshal(resp.Data, &data); err != nil {
		return nil, fmt.Errorf("parse poll data: %w", err)
	}
	return &data, nil
}

// CreateRemoveBackgroundTask submits a background removal job to Kie AI and returns the task ID.
func CreateRemoveBackgroundTask(cfg Config, req RemoveBackgroundRequest) (string, error) {
	if cfg.APIKey == "" {
		return "", fmt.Errorf("KIE_AI_API_KEY is required")
	}
	body, err := json.Marshal(req)
	if err != nil {
		return "", err
	}
	url := buildURL(cfg.BaseURL, cfg.CreateTaskPath)
	raw, err := doRequest(http.MethodPost, url, cfg.APIKey, body, cfg.TimeoutSeconds, cfg.HTTPRetries, cfg.HTTPRetryBackoff)
	if err != nil {
		return "", err
	}
	var resp APIResponse
	if err := json.Unmarshal(raw, &resp); err != nil {
		return "", fmt.Errorf("parse response: %w", err)
	}
	if resp.Code != 200 {
		return "", fmt.Errorf("API error (code %d): %s", resp.Code, resp.Msg)
	}
	if len(resp.Data) == 0 {
		return "", fmt.Errorf("API returned empty data field")
	}
	var data CreateTaskData
	if err := json.Unmarshal(resp.Data, &data); err != nil {
		return "", fmt.Errorf("parse task data: %w", err)
	}
	if data.TaskID == "" {
		return "", fmt.Errorf("no task ID in response")
	}
	return data.TaskID, nil
}

// UploadFile uploads a local file to Kie AI's file upload service and returns the downloadUrl.
func UploadFile(cfg Config, filePath string) (string, error) {
	if cfg.APIKey == "" {
		return "", fmt.Errorf("KIE_AI_API_KEY is required")
	}
	f, err := os.Open(filePath)
	if err != nil {
		return "", fmt.Errorf("open file: %w", err)
	}
	defer f.Close()

	var buf bytes.Buffer
	mw := multipart.NewWriter(&buf)
	fw, err := mw.CreateFormFile("file", filepath.Base(filePath))
	if err != nil {
		return "", fmt.Errorf("create form file: %w", err)
	}
	if _, err := io.Copy(fw, f); err != nil {
		return "", fmt.Errorf("copy file: %w", err)
	}
	_ = mw.WriteField("uploadPath", "mcp-uploads")
	mw.Close()

	url := strings.TrimRight(cfg.FileUploadURL, "/") + "/api/file-stream-upload"
	client := &http.Client{Timeout: time.Duration(cfg.TimeoutSeconds) * time.Second}
	req, err := http.NewRequest(http.MethodPost, url, &buf)
	if err != nil {
		return "", err
	}
	req.Header.Set("Authorization", "Bearer "+cfg.APIKey)
	req.Header.Set("Content-Type", mw.FormDataContentType())

	resp, err := client.Do(req)
	if err != nil {
		return "", fmt.Errorf("upload request: %w", err)
	}
	defer resp.Body.Close()
	raw, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", fmt.Errorf("read upload response: %w", err)
	}
	if resp.StatusCode >= 300 {
		return "", fmt.Errorf("upload HTTP %d: %s", resp.StatusCode, string(raw))
	}

	var result struct {
		Success bool `json:"success"`
		Code    int  `json:"code"`
		Data    struct {
			DownloadURL string `json:"downloadUrl"`
		} `json:"data"`
		Msg string `json:"msg"`
	}
	if err := json.Unmarshal(raw, &result); err != nil {
		return "", fmt.Errorf("parse upload response: %w", err)
	}
	if !result.Success || result.Data.DownloadURL == "" {
		return "", fmt.Errorf("upload failed: %s", result.Msg)
	}
	return result.Data.DownloadURL, nil
}

// DownloadImage fetches raw image bytes from a URL.
func DownloadImage(url string) ([]byte, error) {
	client := &http.Client{Timeout: 60 * time.Second}
	resp, err := client.Get(url)
	if err != nil {
		return nil, fmt.Errorf("download failed: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("download HTTP %d for %s", resp.StatusCode, url)
	}
	return io.ReadAll(resp.Body)
}
