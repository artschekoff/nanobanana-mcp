BINARY := bin/kie-ai-mcp

.PHONY: build run test clean

build:
	go build -o $(BINARY) .

run:
	go run .

test:
	go test ./...

clean:
	rm -rf bin/
