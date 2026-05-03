package main

import (
	"log"
	"os"
	"path/filepath"

	"github.com/artschekoff/kie-ai-mcp/internal/kieai"
	"github.com/artschekoff/kie-ai-mcp/internal/tools"
	"github.com/joho/godotenv"
	"github.com/mark3labs/mcp-go/server"
)

func main() {
	loadDotEnv()

	cfg := kieai.LoadConfig()
	if cfg.APIKey == "" {
		log.Fatal("KIE_AI_API_KEY environment variable is required")
	}

	s := server.NewMCPServer("kie-ai-mcp", "1.0.0")
	tools.Register(s, cfg)

	if err := server.ServeStdio(s); err != nil {
		log.Fatalf("MCP server error: %v", err)
	}
}

func loadDotEnv() {
	_ = godotenv.Load(".env")
	if exe, err := os.Executable(); err == nil {
		_ = godotenv.Load(filepath.Join(filepath.Dir(exe), ".env"))
	}
}
