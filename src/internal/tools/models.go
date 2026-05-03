package tools

import (
	"context"

	"github.com/mark3labs/mcp-go/mcp"
)

type modelInfo struct {
	ID   string `json:"id"`
	Type string `json:"type"`
}

var availableModels = []modelInfo{
	{ID: "gpt-image-2-text-to-image", Type: "text-to-image"},
	{ID: "gpt-image/1.5-text-to-image", Type: "text-to-image"},
	{ID: "nano-banana-2", Type: "text-to-image"},
	{ID: "nano-banana-pro", Type: "text-to-image"},
	{ID: "qwen/text-to-image", Type: "text-to-image"},
	{ID: "qwen2/text-to-image", Type: "text-to-image"},
	{ID: "grok-imagine/text-to-image", Type: "text-to-image"},
	{ID: "z-image", Type: "text-to-image"},
}

func listModelsHandler() func(context.Context, mcp.CallToolRequest) (*mcp.CallToolResult, error) {
	return func(_ context.Context, _ mcp.CallToolRequest) (*mcp.CallToolResult, error) {
		return jsonResult(map[string]any{
			"models": availableModels,
		})
	}
}
