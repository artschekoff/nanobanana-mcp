package kieai

import (
	"encoding/json"
	"os"
	"strconv"
)

// Config holds all runtime configuration loaded from environment variables.
type Config struct {
	APIKey           string
	BaseURL          string
	FileUploadURL    string
	Model            string
	TimeoutSeconds   int
	PollIntervalSec  float64
	PollTimeoutSec   int
	CreateTaskPath   string
	GetTaskPath      string
	OutputDir        string
	HTTPRetries      int
	HTTPRetryBackoff float64
}

// LoadConfig reads configuration from environment variables.
func LoadConfig() Config {
	return Config{
		APIKey:           os.Getenv("KIE_AI_API_KEY"),
		BaseURL:          envOr("KIE_AI_BASE_URL", "https://api.kie.ai/api/v1"),
		FileUploadURL:    envOr("KIE_AI_FILE_UPLOAD_URL", "https://kieai.redpandaai.co"),
		Model:            envOr("KIE_AI_MODEL", "nano-banana-pro"),
		TimeoutSeconds:   envInt("KIE_AI_TIMEOUT_SECONDS", 90),
		PollIntervalSec:  envFloat("KIE_AI_POLL_INTERVAL_SECONDS", 3.0),
		PollTimeoutSec:   envInt("KIE_AI_POLL_TIMEOUT_SECONDS", 300),
		CreateTaskPath:   envOr("KIE_AI_CREATE_TASK_PATH", "/jobs/createTask"),
		GetTaskPath:      envOr("KIE_AI_GET_TASK_PATH", "/jobs/recordInfo"),
		OutputDir:        envOr("KIE_AI_OUTPUT_DIR", "output"),
		HTTPRetries:      envInt("KIE_AI_HTTP_RETRIES", 3),
		HTTPRetryBackoff: envFloat("KIE_AI_HTTP_RETRY_BACKOFF_SECONDS", 1.5),
	}
}

func envOr(key, def string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return def
}

func envInt(key string, def int) int {
	if v := os.Getenv(key); v != "" {
		if n, err := strconv.Atoi(v); err == nil {
			return n
		}
	}
	return def
}

func envFloat(key string, def float64) float64 {
	if v := os.Getenv(key); v != "" {
		if f, err := strconv.ParseFloat(v, 64); err == nil {
			return f
		}
	}
	return def
}

// CreateTaskRequest is the body for POST /jobs/createTask.
type CreateTaskRequest struct {
	Model       string    `json:"model"`
	Input       TaskInput `json:"input"`
	CallBackURL string    `json:"callBackUrl,omitempty"`
}

// TaskInput describes the image generation parameters.
type TaskInput struct {
	Prompt       string   `json:"prompt"`
	ImageInput   []string `json:"image_input"`
	AspectRatio  string   `json:"aspect_ratio"`
	Resolution   string   `json:"resolution"`
	OutputFormat string   `json:"output_format"`
	Platform     string   `json:"platform,omitempty"`
}

// APIResponse is the standard Kie AI envelope.
type APIResponse struct {
	Code int             `json:"code"`
	Msg  string          `json:"msg"`
	Data json.RawMessage `json:"data"`
}

// CreateTaskData is the payload inside APIResponse.Data for createTask.
type CreateTaskData struct {
	TaskID string `json:"taskId"`
}

// PollData is the payload inside APIResponse.Data for recordInfo.
type PollData struct {
	State      string `json:"state"`
	ResultJSON string `json:"resultJson"`
	FailMsg    string `json:"failMsg"`
}

// ResultJSON is the parsed content of PollData.ResultJSON.
type ResultJSON struct {
	ResultURLs []string `json:"resultUrls"`
}

// RemoveBackgroundRequest is the body for POST /jobs/createTask with recraft/remove-background model.
type RemoveBackgroundRequest struct {
	Model       string                `json:"model"`
	CallBackURL string                `json:"callBackUrl,omitempty"`
	Input       RemoveBackgroundInput `json:"input"`
}

// RemoveBackgroundInput describes the background removal parameters.
type RemoveBackgroundInput struct {
	Image       string `json:"image"`
	NSFWChecker bool   `json:"nsfw_checker"`
}
