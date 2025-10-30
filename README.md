# Agent Stack Showcase Agent 🤖💬🧪

The **Agent Stack Showcase Agent** is a research prototype built with the [BeeAI Framework](https://framework.beeai.dev/) and [Agent Stack SDK](https://docs.beeai.dev/).

It demonstrates how to combine tool orchestration, memory, file analysis, and platform extensions into a general-purpose conversational assistant. The agent can handle chat, process uploaded files, search the web, and provide structured outputs with citations and trajectory logs for debugging and UI replay.

---

## ✨ Capabilities

* **Multi-turn chat** with persistent per-session memory (`UnconstrainedMemory`)
* **Tool orchestration** via the experimental `RequirementAgent`, with rules like:

  * `ThinkTool` — invoked first and after every tool for reasoning
  * `DuckDuckGoSearchTool` — used up to 2 times per query, skipped for casual greetings
  * **File processing** — supports PDF, CSV, JSON, and plain text uploads
* **Citation extraction** — converts `[text](url)` markdown links into structured citation objects
* **Trajectory tracking** — logs each reasoning step, tool invocation, and output for replay/debugging
* **Configurable settings** — users can toggle thinking/search behaviors and select response style (concise, standard, detailed)
* **Basic error handling** — user-facing messages and detailed logs

---

## 🚀 Running the Agent

1. **Install Agent Stack**
   Follow the [Quickstart Guide](https://docs.beeai.dev/introduction/quickstart) to install and set up Agent Stack.
   This is required before running the agent.

2. **Start the server**
   Once the platform is installed, launch the agent server:

   ```bash
   uv run server
   ```

   The server runs on the configured `HOST` and `PORT` environment variables (defaults: `127.0.0.1:8000`).

---

## 🧩 Key Components

* **`agentstack_showcase(...)`** — Main async entrypoint handling chat, file uploads, memory, and tool orchestration
* **`RequirementAgent(...)`** — Experimental agent that enforces `ConditionalRequirement` rules for tool usage
* **`ThinkTool`** — Provides structured reasoning and analysis
* **`DuckDuckGoSearchTool`** — Performs real-time web search (with constraints)
* **`extract_citations(...)`** — Converts markdown links into structured citation objects
* **`is_casual(...)`** — Skips tool invocation for short greetings or casual input
* **`get_memory(...)`** — Provides per-session `UnconstrainedMemory`
* **`run()`** — Starts the Agent Stack server

---

## 🔌 Extensions

* **CitationExtensionServer** — renders citations into structured previews
* **TrajectoryExtensionServer** — captures reasoning/tool usage for UI replay & debugging
* **LLMServiceExtensionServer** — manages LLM fulfillment through Agent Stack
* **SettingsExtensionServer** — allows user configuration of agent behaviors

---

## 💡 Example Interaction

**User input**:

> What are the latest advancements in AI research from 2025?

**Agent flow**:

1. `ThinkTool` invoked for reasoning
2. `DuckDuckGoSearchTool` called (unless skipped for casual input)
3. Response returned with proper `[label](url)` citations
4. Citations extracted and sent to UI
5. Steps logged in trajectory extension
6. Conversation context persisted for future turns
7. If a file is uploaded, it’s analyzed and summarized

---

## 📂 Example Skills

The agent supports both **chat** and **file analysis**, such as:

* "What are the latest advancements in AI research from 2025?"
* "Can you help me write a Slack announcement for \[topic/team update]?"
* "Analyze this CSV file and tell me the key trends."
* "Summarize the main points from this PDF document."
