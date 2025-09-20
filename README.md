# BeeAI Showcase Agent 🤖💬🧪

This is a research prototype of a general-purpose chat assistant built with the [BeeAI Framework](https://framework.beeai.dev/) and [BeeAI SDK](https://docs.beeai.dev/). It demonstrates advanced tool orchestration with the experimental `RequirementAgent` pattern, platform extensions, and support for both chat and file analysis.

The assistant runs as a server and supports interactive use with tool-based reasoning, retrieval, memory, structured metadata, and UI integrations.

## Capabilities

- Streaming multi-turn chat with persistent session memory
- Tool orchestration via `RequirementAgent` with conditional rules:
    - `ThinkTool` — always invoked at step 1 and after any tool
    - `DuckDuckGoSearchTool` — max 2 invocations per query, skipped for casual/greeting messages
    - File Processing — supports PDF, CSV, JSON, and plain text uploads
- Session memory with `UnconstrainedMemory`
- Citation extraction from markdown-style links into structured objects
- Trajectory tracking — logs each reasoning step, tool call, and output for UI replay/debugging
- Basic error handling with visible logs and user-facing error messages

## Running the Agent

To start the server:

```
uv run server
```

The server runs on the configured HOST and PORT environment variables (defaults: 127.0.0.1:8000).

## Key Functions & Components

- `general_chat_assistant(...)`: Main async agent entrypoint (handles chat, tools, memory, file analysis)
- `RequirementAgent(...)`: Orchestrates tools with ConditionalRequirement rules
- `extract_citations(...)`: Converts markdown [text](url) links into structured citation objects
- `is_casual(...)`: Detects short casual messages to skip tool invocation
- `get_memory(...)`: Provides per-session UnconstrainedMemory
- `run()`: Starts the BeeAI server

## Extensions

- `CitationExtensionServer` — renders citations from [text](url) into structured link previews
- `TrajectoryExtensionServer` — captures reasoning/tool usage for UI replay and debugging
- `LLMServiceExtensionServer` — manages LLM fulfillment via BeeAI platform

## Example

**Input**:

> What are the latest advancements in AI research from 2025?

**Result**:

- Invokes `ThinkTool` for reasoning
- Calls `DuckDuckGoSearchTool` with a relevant query (unless skipped for casual)
- Returns a final response with proper [label](url) citations
- Extracted citations sent to UI for rendering
- All steps logged via trajectory extension
- Conversation context persisted across turns
- If a file is uploaded, analyzes and summarizes its contents
