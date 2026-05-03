package tools

import (
	"context"
	"fmt"

	"github.com/artschekoff/kie-ai-mcp/internal/kieai"
	"github.com/mark3labs/mcp-go/mcp"
)

func describeHandler(cfg kieai.Config) func(context.Context, mcp.CallToolRequest) (*mcp.CallToolResult, error) {
	return func(_ context.Context, _ mcp.CallToolRequest) (*mcp.CallToolResult, error) {
		return jsonResult(map[string]any{
			"server": "kie-ai-mcp",
			"api": map[string]string{
				"create_task": fmt.Sprintf("%s%s", cfg.BaseURL, cfg.CreateTaskPath),
				"get_task":    fmt.Sprintf("%s%s", cfg.BaseURL, cfg.GetTaskPath),
			},
			"async_flow": "submit task → poll get_task until state=success → image in resultUrls[0]",
			"tools": []string{
				"describe_imager_interface",
				"create_visual_task",
				"get_visual_task",
				"generate_visual",
				"generate_visual_batch",
			},
			"environment": map[string]string{
				"KIE_AI_API_KEY":                    "required",
				"KIE_AI_BASE_URL":                   "optional (default: https://api.kie.ai/api/v1)",
				"KIE_AI_MODEL":                      "optional (default: nano-banana-pro)",
				"KIE_AI_TIMEOUT_SECONDS":            "optional (default: 90)",
				"KIE_AI_POLL_INTERVAL_SECONDS":      "optional (default: 3)",
				"KIE_AI_POLL_TIMEOUT_SECONDS":       "optional (default: 300)",
				"KIE_AI_OUTPUT_DIR":                 "optional (default: output)",
				"KIE_AI_HTTP_RETRIES":               "optional (default: 3)",
				"KIE_AI_HTTP_RETRY_BACKOFF_SECONDS": "optional (default: 1.5)",
			},
		})
	}
}
