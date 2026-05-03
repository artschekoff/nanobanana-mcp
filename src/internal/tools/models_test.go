package tools

import (
	"context"
	"encoding/json"
	"testing"

	"github.com/mark3labs/mcp-go/mcp"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestListModelsHandler_ReturnsAllModels(t *testing.T) {
	handler := listModelsHandler()
	result, err := handler(context.Background(), mcp.CallToolRequest{})
	require.NoError(t, err)
	require.NotNil(t, result)
	require.False(t, result.IsError)
	require.Len(t, result.Content, 1)

	text := result.Content[0].(mcp.TextContent).Text
	var resp struct {
		Models []struct {
			ID   string `json:"id"`
			Type string `json:"type"`
		} `json:"models"`
	}
	require.NoError(t, json.Unmarshal([]byte(text), &resp))

	assert.Len(t, resp.Models, 8)

	ids := make(map[string]bool)
	for _, m := range resp.Models {
		ids[m.ID] = true
		assert.Equal(t, "text-to-image", m.Type)
	}

	assert.True(t, ids["gpt-image-2-text-to-image"])
	assert.True(t, ids["gpt-image/1.5-text-to-image"])
	assert.True(t, ids["nano-banana-2"])
	assert.True(t, ids["nano-banana-pro"])
	assert.True(t, ids["qwen/text-to-image"])
	assert.True(t, ids["qwen2/text-to-image"])
	assert.True(t, ids["grok-imagine/text-to-image"])
	assert.True(t, ids["z-image"])
}
