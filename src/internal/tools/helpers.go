package tools

import (
	"encoding/json"
	"fmt"
	"strings"
	"time"

	"github.com/mark3labs/mcp-go/mcp"
)

func argStr(req mcp.CallToolRequest, key, def string) string {
	args := req.GetArguments()
	if v, ok := args[key]; ok {
		if s, ok := v.(string); ok && s != "" {
			return s
		}
	}
	return def
}

func argBool(req mcp.CallToolRequest, key string, def bool) bool {
	args := req.GetArguments()
	if v, ok := args[key]; ok {
		if b, ok := v.(bool); ok {
			return b
		}
	}
	return def
}

func argFloat(req mcp.CallToolRequest, key string, def float64) float64 {
	args := req.GetArguments()
	if v, ok := args[key]; ok {
		switch n := v.(type) {
		case float64:
			return n
		case int:
			return float64(n)
		}
	}
	return def
}

func argInt(req mcp.CallToolRequest, key string, def int) int {
	return int(argFloat(req, key, float64(def)))
}

// argSlice extracts an array argument, accepting both []any (JSON array) and
// string (JSON-encoded array).
func argSlice(req mcp.CallToolRequest, key string) ([]any, bool) {
	args := req.GetArguments()
	v, ok := args[key]
	if !ok {
		return nil, false
	}
	switch val := v.(type) {
	case []any:
		return val, true
	case string:
		var items []any
		if err := json.Unmarshal([]byte(val), &items); err == nil {
			return items, true
		}
	}
	return nil, false
}

func jsonResult(v any) (*mcp.CallToolResult, error) {
	data, err := json.Marshal(v)
	if err != nil {
		return mcp.NewToolResultError(fmt.Sprintf("marshal result: %v", err)), nil
	}
	return mcp.NewToolResultText(string(data)), nil
}

func composePrompt(style, prompt, platform, negativePrompt string) string {
	var b strings.Builder
	b.WriteString("Follow this visual style strictly:\n")
	b.WriteString(strings.TrimSpace(style))
	if p := strings.TrimSpace(platform); p != "" {
		b.WriteString("\n\nTarget social platform:\n")
		b.WriteString(p)
	}
	b.WriteString("\n\nGenerate this scene:\n")
	b.WriteString(strings.TrimSpace(prompt))
	if n := strings.TrimSpace(negativePrompt); n != "" {
		b.WriteString("\n\nAvoid:\n")
		b.WriteString(n)
	}
	return b.String()
}

func defaultFileName(prompt, ext string) string {
	ts := time.Now().UTC().Format("20060102-150405")
	return ts + "-" + slugify(prompt, 48) + "." + ext
}

func slugify(s string, maxLen int) string {
	var b strings.Builder
	prevDash := false
	for _, r := range strings.ToLower(s) {
		if (r >= 'a' && r <= 'z') || (r >= '0' && r <= '9') {
			b.WriteRune(r)
			prevDash = false
		} else if !prevDash {
			b.WriteRune('-')
			prevDash = true
		}
		if b.Len() >= maxLen {
			break
		}
	}
	result := strings.Trim(b.String(), "-")
	if result == "" {
		return "visual"
	}
	return result
}

func normalizeFormat(f string) string {
	f = strings.ToLower(strings.TrimSpace(f))
	if f == "jpg" {
		return "jpeg"
	}
	switch f {
	case "png", "jpeg", "webp":
		return f
	}
	return "png"
}

func truncate(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n]
}
