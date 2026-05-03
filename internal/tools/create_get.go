package tools

import (
	"context"
	"encoding/json"
	"strings"

	"github.com/artschekoff/kie-ai-mcp/internal/kieai"
	"github.com/mark3labs/mcp-go/mcp"
)

func createVisualTaskHandler(cfg kieai.Config) func(context.Context, mcp.CallToolRequest) (*mcp.CallToolResult, error) {
	return func(_ context.Context, req mcp.CallToolRequest) (*mcp.CallToolResult, error) {
		style := argStr(req, "style", "")
		prompt := argStr(req, "prompt", "")
		if style == "" {
			return mcp.NewToolResultError("'style' is required"), nil
		}
		if prompt == "" {
			return mcp.NewToolResultError("'prompt' is required"), nil
		}

		platform := argStr(req, "platform", "")
		model := argStr(req, "model", cfg.Model)
		aspectRatio := argStr(req, "aspect_ratio", "1:1")
		resolution := argStr(req, "resolution", "1K")
		outFmt := normalizeFormat(argStr(req, "output_format", "png"))
		negativePrompt := argStr(req, "negative_prompt", "")
		callbackURL := argStr(req, "callback_url", "")

		finalPrompt := composePrompt(style, prompt, platform, negativePrompt)
		createReq := kieai.CreateTaskRequest{
			Model:       model,
			CallBackURL: callbackURL,
			Input: kieai.TaskInput{
				Prompt:       finalPrompt,
				ImageInput:   []string{},
				AspectRatio:  aspectRatio,
				Resolution:   resolution,
				OutputFormat: outFmt,
				Platform:     platform,
			},
		}

		taskID, err := kieai.CreateTask(cfg, createReq)
		if err != nil {
			return mcp.NewToolResultError(err.Error()), nil
		}

		return jsonResult(map[string]any{
			"ok":             true,
			"task_id":        taskID,
			"model":          model,
			"platform":       platform,
			"output_format":  outFmt,
			"aspect_ratio":   aspectRatio,
			"resolution":     resolution,
			"prompt_preview": truncate(finalPrompt, 300),
		})
	}
}

func getVisualTaskHandler(cfg kieai.Config) func(context.Context, mcp.CallToolRequest) (*mcp.CallToolResult, error) {
	return func(_ context.Context, req mcp.CallToolRequest) (*mcp.CallToolResult, error) {
		taskID := strings.TrimSpace(argStr(req, "task_id", ""))
		if taskID == "" {
			return mcp.NewToolResultError("'task_id' is required"), nil
		}

		data, err := kieai.GetTask(cfg, taskID)
		if err != nil {
			return mcp.NewToolResultError(err.Error()), nil
		}

		result := map[string]any{
			"ok":      true,
			"task_id": taskID,
			"status":  data.State,
		}
		if data.ResultJSON != "" {
			var rj kieai.ResultJSON
			if jsonErr := json.Unmarshal([]byte(data.ResultJSON), &rj); jsonErr == nil && len(rj.ResultURLs) > 0 {
				result["image_urls"] = rj.ResultURLs
			}
		}
		if data.FailMsg != "" {
			result["fail_msg"] = data.FailMsg
		}

		return jsonResult(result)
	}
}
