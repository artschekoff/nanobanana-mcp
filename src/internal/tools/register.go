package tools

import (
	"github.com/artschekoff/kie-ai-mcp/src/internal/kieai"
	"github.com/mark3labs/mcp-go/mcp"
	"github.com/mark3labs/mcp-go/server"
)

func Register(s *server.MCPServer, cfg kieai.Config) {
	s.AddTool(
		mcp.NewTool("describe_imager_interface",
			mcp.WithDescription("Return tool contracts and environment config for kie-ai-mcp."),
		),
		describeHandler(cfg),
	)

	s.AddTool(
		mcp.NewTool("list_models",
			mcp.WithDescription("Return the list of available image generation models. Use the returned id as the model parameter in other tools."),
		),
		listModelsHandler(),
	)

	s.AddTool(
		mcp.NewTool("create_visual_task",
			mcp.WithDescription("Submit an image generation job to Kie AI. Returns task_id immediately (async). Poll with get_visual_task."),
			mcp.WithString("style", mcp.Required(), mcp.Description("Visual style description")),
			mcp.WithString("prompt", mcp.Required(), mcp.Description("Scene to generate")),
			mcp.WithString("platform", mcp.Description("Target social platform")),
			mcp.WithString("model", mcp.Description("Model name")),
			mcp.WithString("aspect_ratio", mcp.Description("1:1, 16:9, 9:16, 4:3 (default: 1:1)")),
			mcp.WithString("resolution", mcp.Description("1K, 2K (default: 1K)")),
			mcp.WithString("output_format", mcp.Description("png, jpeg, webp (default: png)")),
			mcp.WithString("negative_prompt", mcp.Description("Elements to exclude")),
			mcp.WithString("callback_url", mcp.Description("Webhook URL for completion")),
		),
		createVisualTaskHandler(cfg),
	)

	s.AddTool(
		mcp.NewTool("get_visual_task",
			mcp.WithDescription("Check status of a Kie AI image task. Returns status and image_urls when complete."),
			mcp.WithString("task_id", mcp.Required(), mcp.Description("Task ID from create_visual_task")),
		),
		getVisualTaskHandler(cfg),
	)

	s.AddTool(
		mcp.NewTool("generate_visual",
			mcp.WithDescription("Generate an image end-to-end: submit, poll queue, download, save file. Use wait_for_result=false to get task_id immediately and poll manually."),
			mcp.WithString("style", mcp.Required(), mcp.Description("Visual style description")),
			mcp.WithString("prompt", mcp.Required(), mcp.Description("Scene to generate")),
			mcp.WithString("platform", mcp.Description("Target social platform")),
			mcp.WithString("output_path", mcp.Description("Output file path or directory")),
			mcp.WithString("model", mcp.Description("Model name")),
			mcp.WithString("aspect_ratio", mcp.Description("Aspect ratio (default: 1:1)")),
			mcp.WithString("resolution", mcp.Description("Resolution (default: 1K)")),
			mcp.WithString("output_format", mcp.Description("png, jpeg, webp (default: png)")),
			mcp.WithString("negative_prompt", mcp.Description("Elements to exclude")),
			mcp.WithString("callback_url", mcp.Description("Webhook URL")),
			mcp.WithBoolean("wait_for_result", mcp.Description("Poll until done and save (default: true)")),
			mcp.WithNumber("poll_interval_seconds", mcp.Description("Seconds between polls (default: 3)")),
			mcp.WithNumber("poll_timeout_seconds", mcp.Description("Max seconds to wait (default: 300)")),
		),
		generateVisualHandler(cfg),
	)

	s.AddTool(
		mcp.NewTool("create_remove_bg_task",
			mcp.WithDescription("Submit a background removal job to Kie AI. Returns task_id immediately (async). Poll with get_remove_bg_task."),
			mcp.WithString("image", mcp.Required(), mcp.Description("Image URL or local file path (PNG, JPG, WEBP, max 5MB)")),
			mcp.WithBoolean("nsfw_checker", mcp.Description("Enable NSFW check (default: true)")),
			mcp.WithString("callback_url", mcp.Description("Webhook URL for completion")),
		),
		createRemoveBgTaskHandler(cfg),
	)

	s.AddTool(
		mcp.NewTool("get_remove_bg_task",
			mcp.WithDescription("Check status of a background removal task. Returns image_url when done."),
			mcp.WithString("task_id", mcp.Required(), mcp.Description("Task ID from create_remove_bg_task")),
		),
		getVisualTaskHandler(cfg),
	)

	s.AddTool(
		mcp.NewTool("remove_background",
			mcp.WithDescription("Remove image background end-to-end: submit, poll, download, save file. Use wait_for_result=false to get task_id and poll manually."),
			mcp.WithString("image", mcp.Required(), mcp.Description("Image URL or local file path (PNG, JPG, WEBP, max 5MB)")),
			mcp.WithBoolean("nsfw_checker", mcp.Description("Enable NSFW check (default: true)")),
			mcp.WithString("output_path", mcp.Description("Output file path or directory")),
			mcp.WithBoolean("wait_for_result", mcp.Description("Poll until done and save (default: true)")),
			mcp.WithNumber("poll_interval_seconds", mcp.Description("Seconds between polls (default: 3)")),
			mcp.WithNumber("poll_timeout_seconds", mcp.Description("Max seconds to wait (default: 300)")),
			mcp.WithString("callback_url", mcp.Description("Webhook URL for completion")),
		),
		removeBackgroundHandler(cfg),
	)

	s.AddTool(
		mcp.NewTool("generate_visual_batch",
			mcp.WithDescription("Generate multiple images in parallel. Each item needs 'prompt' and 'style'."),
			mcp.WithString("items", mcp.Required(), mcp.Description("JSON array of generation items")),
			mcp.WithString("default_style", mcp.Description("Fallback style")),
			mcp.WithString("default_platform", mcp.Description("Fallback platform")),
			mcp.WithNumber("max_workers", mcp.Description("Max parallel generations (default: 3)")),
			mcp.WithBoolean("continue_on_error", mcp.Description("Continue on item failure (default: true)")),
			mcp.WithBoolean("default_wait_for_result", mcp.Description("Wait for each result (default: true)")),
			mcp.WithNumber("default_poll_interval_seconds", mcp.Description("Poll interval")),
			mcp.WithNumber("default_poll_timeout_seconds", mcp.Description("Poll timeout")),
		),
		generateVisualBatchHandler(cfg),
	)
}
