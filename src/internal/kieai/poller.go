package kieai

import (
	"encoding/json"
	"fmt"
	"time"
)

var successStates = map[string]bool{
	"success": true, "succeeded": true, "completed": true,
	"done": true, "finish": true, "finished": true,
}

var failStates = map[string]bool{
	"failed": true, "error": true, "cancelled": true,
	"canceled": true, "timeout": true, "fail": true,
}

// PollUntilDone polls GetTask in a loop until the task reaches a terminal state.
// Returns the list of image URLs on success.
func PollUntilDone(cfg Config, taskID string) ([]string, error) {
	deadline := time.Now().Add(time.Duration(cfg.PollTimeoutSec) * time.Second)
	interval := time.Duration(cfg.PollIntervalSec * float64(time.Second))
	if interval < 50*time.Millisecond {
		interval = 50 * time.Millisecond
	}

	for time.Now().Before(deadline) {
		data, err := GetTask(cfg, taskID)
		if err != nil {
			return nil, fmt.Errorf("poll error: %w", err)
		}

		if failStates[data.State] {
			msg := data.FailMsg
			if msg == "" {
				msg = "task failed with state: " + data.State
			}
			return nil, fmt.Errorf("task failed: %s", msg)
		}

		if successStates[data.State] {
			return extractResultURLs(data.ResultJSON)
		}

		time.Sleep(interval)
	}

	return nil, fmt.Errorf("task %s did not complete within %ds", taskID, cfg.PollTimeoutSec)
}

func extractResultURLs(resultJSON string) ([]string, error) {
	if resultJSON == "" {
		return nil, fmt.Errorf("completed task has no resultJson")
	}
	var r ResultJSON
	if err := json.Unmarshal([]byte(resultJSON), &r); err != nil {
		return nil, fmt.Errorf("parse resultJson: %w", err)
	}
	if len(r.ResultURLs) == 0 {
		return nil, fmt.Errorf("no result URLs in completed task")
	}
	return r.ResultURLs, nil
}
