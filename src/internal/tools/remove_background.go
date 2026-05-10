package tools

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/artschekoff/kie-ai-mcp/src/internal/kieai"
	mcp "github.com/mark3labs/mcp-go/mcp"
)

const removeBgModel = "recraft/remove-background"

func createRemoveBgTaskHandler(cfg kieai.Config) func(context.Context, mcp.CallToolRequest) (*mcp.CallToolResult, error) {
	return func(_ context.Context, req mcp.CallToolRequest) (*mcp.CallToolResult, error) {
		image := argStr(req, "image", "")
		if image == "" {
			return mcp.NewToolResultError("'image' is required"), nil
		}
		nsfwChecker := argBool(req, "nsfw_checker", true)
		callbackURL := argStr(req, "callback_url", "")

		imageData, err := resolveImage(cfg, image)
		if err != nil {
			return mcp.NewToolResultError(err.Error()), nil
		}

		taskID, err := kieai.CreateRemoveBackgroundTask(cfg, kieai.RemoveBackgroundRequest{
			Model:       removeBgModel,
			CallBackURL: callbackURL,
			Input:       kieai.RemoveBackgroundInput{Image: imageData, NSFWChecker: nsfwChecker},
		})
		if err != nil {
			return mcp.NewToolResultError(err.Error()), nil
		}

		return jsonResult(map[string]any{
			"ok":      true,
			"task_id": taskID,
			"model":   removeBgModel,
		})
	}
}

func removeBackgroundHandler(cfg kieai.Config) func(context.Context, mcp.CallToolRequest) (*mcp.CallToolResult, error) {
	return func(_ context.Context, req mcp.CallToolRequest) (*mcp.CallToolResult, error) {
		image := argStr(req, "image", "")
		if image == "" {
			return mcp.NewToolResultError("'image' is required"), nil
		}
		nsfwChecker := argBool(req, "nsfw_checker", true)
		callbackURL := argStr(req, "callback_url", "")
		outputPath := argStr(req, "output_path", "")
		waitForResult := argBool(req, "wait_for_result", true)
		pollInterval := argFloat(req, "poll_interval_seconds", cfg.PollIntervalSec)
		pollTimeout := argInt(req, "poll_timeout_seconds", cfg.PollTimeoutSec)

		imageData, err := resolveImage(cfg, image)
		if err != nil {
			return mcp.NewToolResultError(err.Error()), nil
		}

		taskID, err := kieai.CreateRemoveBackgroundTask(cfg, kieai.RemoveBackgroundRequest{
			Model:       removeBgModel,
			CallBackURL: callbackURL,
			Input:       kieai.RemoveBackgroundInput{Image: imageData, NSFWChecker: nsfwChecker},
		})
		if err != nil {
			return mcp.NewToolResultError(fmt.Sprintf("create task: %v", err)), nil
		}

		if !waitForResult {
			return jsonResult(map[string]any{
				"ok":      true,
				"task_id": taskID,
				"status":  "submitted",
				"hint":    "Call get_remove_bg_task with this task_id to check status",
			})
		}

		pollCfg := cfg
		pollCfg.PollIntervalSec = pollInterval
		pollCfg.PollTimeoutSec = pollTimeout

		imageURLs, err := kieai.PollUntilDone(pollCfg, taskID)
		if err != nil {
			return mcp.NewToolResultError(fmt.Sprintf("poll task %s: %v", taskID, err)), nil
		}

		imgBytes, err := kieai.DownloadImage(imageURLs[0])
		if err != nil {
			return mcp.NewToolResultError(fmt.Sprintf("download image: %v", err)), nil
		}

		filePath := resolveOutputPath(outputPath, cfg.OutputDir, "remove-bg", "png")
		if err := os.MkdirAll(filepath.Dir(filePath), 0o755); err != nil {
			return mcp.NewToolResultError(fmt.Sprintf("create output dir: %v", err)), nil
		}
		if err := os.WriteFile(filePath, imgBytes, 0o644); err != nil {
			return mcp.NewToolResultError(fmt.Sprintf("write image: %v", err)), nil
		}

		return jsonResult(map[string]any{
			"ok":          true,
			"task_id":     taskID,
			"status":      "success",
			"output_path": filePath,
			"image_url":   imageURLs[0],
			"bytes":       len(imgBytes),
		})
	}
}

// resolveImage returns the image as a URL, uploading local files via Kie AI File Upload API.
func resolveImage(cfg kieai.Config, input string) (string, error) {
	if strings.HasPrefix(input, "http://") || strings.HasPrefix(input, "https://") {
		return input, nil
	}
	info, err := os.Stat(input)
	if err != nil {
		return "", fmt.Errorf("read image file: %w", err)
	}
	const maxSize = 5 * 1024 * 1024
	if info.Size() > maxSize {
		return "", fmt.Errorf("image file exceeds 5MB limit (%d bytes)", info.Size())
	}
	return kieai.UploadFile(cfg, input)
}
