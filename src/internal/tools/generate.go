package tools

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"sync"

	"github.com/artschekoff/kie-ai-mcp/src/internal/kieai"
	"github.com/mark3labs/mcp-go/mcp"
)

func generateVisualHandler(cfg kieai.Config) func(context.Context, mcp.CallToolRequest) (*mcp.CallToolResult, error) {
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
		outputPath := argStr(req, "output_path", "")
		waitForResult := argBool(req, "wait_for_result", true)
		pollInterval := argFloat(req, "poll_interval_seconds", cfg.PollIntervalSec)
		pollTimeout := argInt(req, "poll_timeout_seconds", cfg.PollTimeoutSec)

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
			return mcp.NewToolResultError(fmt.Sprintf("create task: %v", err)), nil
		}

		if !waitForResult {
			return jsonResult(map[string]any{
				"ok":       true,
				"task_id":  taskID,
				"status":   "submitted",
				"platform": platform,
				"hint":     "Call get_visual_task with this task_id to check status",
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

		filePath := resolveOutputPath(outputPath, cfg.OutputDir, prompt, outFmt)
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
			"platform":    platform,
			"output_path": filePath,
			"image_url":   imageURLs[0],
			"bytes":       len(imgBytes),
		})
	}
}

func resolveOutputPath(outputPath, outputDir, prompt, ext string) string {
	fileName := defaultFileName(prompt, ext)
	if outputPath == "" {
		dir := outputDir
		if dir == "" {
			dir = "output"
		}
		return filepath.Join(dir, fileName)
	}
	info, err := os.Stat(outputPath)
	if err == nil && info.IsDir() {
		return filepath.Join(outputPath, fileName)
	}
	if len(outputPath) > 0 && (outputPath[len(outputPath)-1] == '/' || outputPath[len(outputPath)-1] == '\\') {
		return filepath.Join(outputPath, fileName)
	}
	return outputPath
}

func generateVisualBatchHandler(cfg kieai.Config) func(context.Context, mcp.CallToolRequest) (*mcp.CallToolResult, error) {
	return func(_ context.Context, req mcp.CallToolRequest) (*mcp.CallToolResult, error) {
		items, ok := argSlice(req, "items")
		if !ok || len(items) == 0 {
			return mcp.NewToolResultError("'items' must be a non-empty array"), nil
		}

		defaultStyle := argStr(req, "default_style", "")
		defaultPlatform := argStr(req, "default_platform", "")
		maxWorkers := argInt(req, "max_workers", 3)
		continueOnError := argBool(req, "continue_on_error", true)
		defaultWait := argBool(req, "default_wait_for_result", true)
		defaultPollInterval := argFloat(req, "default_poll_interval_seconds", cfg.PollIntervalSec)
		defaultPollTimeout := argInt(req, "default_poll_timeout_seconds", cfg.PollTimeoutSec)

		if maxWorkers < 1 {
			maxWorkers = 1
		}
		if maxWorkers > 16 {
			maxWorkers = 16
		}

		type batchResult struct {
			Index  int    `json:"index"`
			OK     bool   `json:"ok"`
			Error  string `json:"error,omitempty"`
			TaskID string `json:"task_id,omitempty"`
			Output string `json:"output_path,omitempty"`
			Bytes  int    `json:"bytes,omitempty"`
			Status string `json:"status,omitempty"`
		}

		results := make([]batchResult, len(items))
		sem := make(chan struct{}, maxWorkers)
		var wg sync.WaitGroup

		for i, rawItem := range items {
			item, ok := rawItem.(map[string]any)
			if !ok {
				results[i] = batchResult{Index: i, OK: false, Error: "item must be an object"}
				continue
			}

			wg.Add(1)
			go func(idx int, it map[string]any) {
				defer wg.Done()
				sem <- struct{}{}
				defer func() { <-sem }()

				sf := func(key, def string) string {
					if v, ok := it[key]; ok {
						if s, ok := v.(string); ok && s != "" {
							return s
						}
					}
					return def
				}

				prompt := sf("prompt", "")
				if prompt == "" {
					results[idx] = batchResult{Index: idx, OK: false, Error: "item missing 'prompt'"}
					return
				}
				style := sf("style", defaultStyle)
				if style == "" {
					results[idx] = batchResult{Index: idx, OK: false, Error: "item missing 'style' and no default_style"}
					return
				}

				platform := sf("platform", defaultPlatform)
				outFmt := normalizeFormat(sf("output_format", "png"))
				aspectRatio := sf("aspect_ratio", "1:1")
				resolution := sf("resolution", "1K")
				model := sf("model", cfg.Model)
				negativePrompt := sf("negative_prompt", "")
				outputPath := sf("output_path", "")

				finalPrompt := composePrompt(style, prompt, platform, negativePrompt)
				createReq := kieai.CreateTaskRequest{
					Model: model,
					Input: kieai.TaskInput{
						Prompt: finalPrompt, ImageInput: []string{},
						AspectRatio: aspectRatio, Resolution: resolution,
						OutputFormat: outFmt, Platform: platform,
					},
				}

				taskID, err := kieai.CreateTask(cfg, createReq)
				if err != nil {
					results[idx] = batchResult{Index: idx, OK: false, Error: err.Error()}
					return
				}

				if !defaultWait {
					results[idx] = batchResult{Index: idx, OK: true, TaskID: taskID, Status: "submitted"}
					return
				}

				pollCfg := cfg
				pollCfg.PollIntervalSec = defaultPollInterval
				pollCfg.PollTimeoutSec = defaultPollTimeout

				imageURLs, err := kieai.PollUntilDone(pollCfg, taskID)
				if err != nil {
					results[idx] = batchResult{Index: idx, OK: false, Error: err.Error(), TaskID: taskID}
					return
				}

				imgBytes, err := kieai.DownloadImage(imageURLs[0])
				if err != nil {
					results[idx] = batchResult{Index: idx, OK: false, Error: err.Error(), TaskID: taskID}
					return
				}

				filePath := resolveOutputPath(outputPath, cfg.OutputDir, prompt, outFmt)
				if mkdirErr := os.MkdirAll(filepath.Dir(filePath), 0o755); mkdirErr != nil {
					results[idx] = batchResult{Index: idx, OK: false, Error: mkdirErr.Error(), TaskID: taskID}
					return
				}
				if writeErr := os.WriteFile(filePath, imgBytes, 0o644); writeErr != nil {
					results[idx] = batchResult{Index: idx, OK: false, Error: writeErr.Error(), TaskID: taskID}
					return
				}

				results[idx] = batchResult{
					Index: idx, OK: true, TaskID: taskID,
					Output: filePath, Bytes: len(imgBytes), Status: "success",
				}
			}(i, item)
		}

		wg.Wait()

		failed := 0
		for _, r := range results {
			if !r.OK {
				failed++
			}
		}

		if failed > 0 && !continueOnError {
			for _, r := range results {
				if !r.OK {
					return mcp.NewToolResultError(r.Error), nil
				}
			}
		}

		return jsonResult(map[string]any{
			"ok":                failed == 0,
			"total":             len(results),
			"succeeded":         len(results) - failed,
			"failed":            failed,
			"max_workers":       maxWorkers,
			"continue_on_error": continueOnError,
			"results":           results,
		})
	}
}
