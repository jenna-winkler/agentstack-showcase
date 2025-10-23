import os
import re
from typing import Annotated
from textwrap import dedent
from dotenv import load_dotenv

from beeai_framework.adapters.openai import OpenAIChatModel
from beeai_framework.backend.types import ChatModelParameters
from beeai_framework.agents.experimental import RequirementAgent
from beeai_framework.agents.experimental.requirements.conditional import ConditionalRequirement
from beeai_framework.agents.requirement.events import RequirementAgentFinalAnswerEvent
from beeai_framework.agents.types import AgentExecutionConfig
from beeai_framework.backend.message import UserMessage, AssistantMessage
from beeai_framework.memory import UnconstrainedMemory
from beeai_framework.tools import Tool
from beeai_framework.tools.search.duckduckgo import DuckDuckGoSearchTool
from beeai_framework.tools.think import ThinkTool
from beeai_framework.emitter import EventMeta

from a2a.types import AgentSkill, Message, Role
from beeai_sdk.server import Server
from beeai_sdk.server.context import RunContext
from beeai_sdk.server.store.platform_context_store import PlatformContextStore
from beeai_sdk.a2a.types import AgentMessage
from beeai_sdk.a2a.extensions import (
    AgentDetail, AgentDetailTool, 
    CitationExtensionServer, CitationExtensionSpec, 
    TrajectoryExtensionServer, TrajectoryExtensionSpec, 
    LLMServiceExtensionServer, LLMServiceExtensionSpec
)
from beeai_sdk.a2a.extensions.ui.settings import (
    CheckboxField,
    CheckboxGroupField,
    SingleSelectField,
    OptionItem,
    SettingsExtensionServer,
    SettingsExtensionSpec,
    SettingsRender,
)
from beeai_sdk.util.file import load_file

load_dotenv()

server = Server()
memories = {}

def get_memory(context: RunContext) -> UnconstrainedMemory:
    """Get or create session memory"""
    context_id = getattr(context, "context_id", getattr(context, "session_id", "default"))
    return memories.setdefault(context_id, UnconstrainedMemory())

def to_framework_message(message: Message):
    """Convert A2A Message to BeeAI Framework Message format"""
    message_text = "".join(part.root.text for part in message.parts if part.root.kind == "text")
    
    if message.role == Role.agent:
        return AssistantMessage(message_text)
    elif message.role == Role.user:
        return UserMessage(message_text)
    else:
        raise ValueError(f"Invalid message role: {message.role}")

def extract_citations(text: str, search_results=None) -> tuple[list[dict], str]:
    """Extract citations and clean text - returns citations in the correct format"""
    citations, offset = [], 0
    pattern = r"\[([^\]]+)\]\(([^)]+)\)"
    
    for match in re.finditer(pattern, text):
        content, url = match.groups()
        start = match.start() - offset

        citations.append({
            "url": url,
            "title": url.split("/")[-1].replace("-", " ").title() or content[:50],
            "description": content[:100] + ("..." if len(content) > 100 else ""),
            "start_index": start, 
            "end_index": start + len(content)
        })
        offset += len(match.group(0)) - len(content)

    return citations, re.sub(pattern, r"\1", text)

def is_casual(msg: str) -> bool:
    """Check if message is casual/greeting"""
    casual_words = {'hey', 'hi', 'hello', 'thanks', 'bye', 'cool', 'nice', 'ok', 'yes', 'no'}
    words = msg.lower().strip().split()
    return len(words) <= 3 and any(w in casual_words for w in words)

@server.agent(
    name="BeeAI Showcase Agent",
    default_input_modes=["text", "text/plain", "application/pdf", "text/csv", "application/json"],
    default_output_modes=["text", "text/plain"],
    detail=AgentDetail(
        interaction_mode="multi-turn",
        user_greeting="Hi! Try out BeeAI features with me — upload a doc, search the web, or tweak my settings.",
        version="0.0.15",
        tools=[
            AgentDetailTool(
                name="Think", 
                description="Advanced reasoning and analysis to provide thoughtful, well-structured responses to complex questions and topics."
            ),
            AgentDetailTool(
                name="DuckDuckGo", 
                description="Search the web for current information, news, and real-time updates on any topic."
            ),
            AgentDetailTool(
                name="File Processing", 
                description="Read and analyze uploaded files including PDFs, text files, CSV data, and JSON documents."
            )
        ],
        framework="BeeAI",
        author={
            "name": "Jenna Winkler"
        },
        source_code_url="https://github.com/jenna-winkler/beeai-showcase-agent"
    ),
    skills=[
        AgentSkill(
            id="beeai-showcase-agent",
            name="BeeAI Showcase Agent",
            description=dedent(
                """\
                The agent is an AI-powered conversational system designed to process user messages, maintain context,
                generate intelligent responses, and analyze uploaded files.
                """
            ),
            tags=["Chat", "Files"],
            examples=[
                "What are the latest advancements in AI research from 2025?",
                "Can you help me write a Slack announcement for [topic/team update]?",
                "Analyze this CSV file and tell me the key trends.",
                "Summarize the main points from this PDF document.",
            ]
        )
    ],
)
async def beeai_showcase_agent(
    input: Message, 
    context: RunContext,
    citation: Annotated[CitationExtensionServer, CitationExtensionSpec()],
    trajectory: Annotated[TrajectoryExtensionServer, TrajectoryExtensionSpec()],
    llm: Annotated[
        LLMServiceExtensionServer, 
        LLMServiceExtensionSpec.single_demand(
            suggested=("meta-llama/llama-3-3-70b-instruct",)
        )
    ],
    settings: Annotated[
        SettingsExtensionServer,
        SettingsExtensionSpec(
            params=SettingsRender(
                fields=[
                    CheckboxGroupField(
                        id="behavior_group",
                        fields=[
                            CheckboxField(
                                id="thinking",
                                label="Thinking",
                                default_value=True,
                            ),
                            CheckboxField(
                                id="search",
                                label="Web Search",
                                default_value=True,
                            )
                        ],
                    ),
                    SingleSelectField(
                        id="response_style",
                        label="Response Style",
                        options=[
                            OptionItem(value="concise", label="Concise Response"),
                            OptionItem(value="standard", label="Standard Response"),
                            OptionItem(value="detailed", label="Detailed Response"),
                        ],
                        default_value="standard",
                    ),
                ],
            ),
        ),
    ],
):
    """
    This is a general-purpose chat assistant prototype built with the BeeAI Framework. It demonstrates advanced capabilities of both the BeeAI Framework and BeeAI SDK.

    ### BeeAI Framework Features

    - **RequirementAgent:** An experimental agent that selects and executes tools based on defined rules instead of relying solely on LLM decisions. ConditionalRequirement rules determine when and how each tool is used.
    - **ThinkTool:** Provides advanced reasoning and structured analysis.
    - **DuckDuckGoSearchTool:** Performs real-time web searches with invocation limits and casual message detection.
    - **Memory Management:** Uses `UnconstrainedMemory` to maintain full conversation context with session persistence.
    - **Streaming Support:** Token-by-token streaming for real-time response feedback.
    - **Error Handling:** Try-catch blocks provide clear messages; `is_casual()` skips unnecessary tool calls for simple messages.

    ### BeeAI SDK Features

    - **GUI Configuration:** Configures agent details including interaction mode, user greeting, tool descriptions, and metadata through AgentDetail.
    - **TrajectoryMetadata:** Logs agent decisions and tool execution for transparency.
    - **CitationMetadata:** Converts markdown links into structured objects for GUI display.
    - **File Processing:** Supports text, PDF, CSV, and JSON files.
    - **LLM Service Extension:** Uses platform-managed LLMs for consistent model access.
    - **Settings Extension:** Allows users to toggle on/off thinking and web search and control response style.
    - **Session History:** Stores messages in platform context for persistent conversation history.
    """
    
    # Store incoming message for session history
    await context.store(input)
    
    thinking_mode = True
    search_enabled = True
    response_style = "standard"

    yield trajectory.trajectory_metadata(
        title="Initializing",
        content="Starting chat assistant and parsing settings"
    )

    if settings:
        try:
            parsed_settings = settings.parse_settings_response()
            
            behavior_group = parsed_settings.values.get("behavior_group")
            if behavior_group and behavior_group.type == "checkbox_group":
                thinking_checkbox = behavior_group.values.get("thinking")
                if thinking_checkbox:
                    thinking_mode = thinking_checkbox.value
                
                search_checkbox = behavior_group.values.get("search")
                if search_checkbox:
                    search_enabled = search_checkbox.value
            
            response_style_field = parsed_settings.values.get("response_style")
            if response_style_field:
                response_style = response_style_field.value
            
            features = []
            if thinking_mode:
                features.append("Thinking")
            if search_enabled:
                features.append("Web Search")
            
            yield trajectory.trajectory_metadata(
                title="Configuration Applied",
                content=f"Features: {', '.join(features) if features else 'Base Chat Only'} | Style: {response_style.title()}"
            )
                
        except Exception as e:
            yield trajectory.trajectory_metadata(
                title="Settings Error",
                content=f"Could not parse settings: {e}, using defaults"
            )

    user_msg = ""
    file_content = ""
    uploaded_files = []
    
    for part in input.parts:
        part_root = part.root
        if part_root.kind == "text":
            user_msg = part_root.text
        elif part_root.kind == "file":
            uploaded_files.append(part_root)
    
    if not user_msg:
        user_msg = "Hello"
    
    memory = get_memory(context)
    
    # Load conversation history into memory
    history = [message async for message in context.load_history() if isinstance(message, Message) and message.parts]
    await memory.add_many(to_framework_message(item) for item in history)
    
    if uploaded_files:
        yield trajectory.trajectory_metadata(
            title="Processing Files",
            content=f"Loading {len(uploaded_files)} uploaded file(s)"
        )
        
        for file_part in uploaded_files:
            try:
                async with load_file(file_part) as loaded_content:
                    filename = file_part.file.name
                    content_type = file_part.file.mime_type
                    content = loaded_content.text
                    file_content += f"\n\n--- File: {filename} ({content_type}) ---\n{content}\n"
                    
                    yield trajectory.trajectory_metadata(
                        title="File Loaded",
                        content=f"Successfully loaded {filename} ({len(content):,} characters)"
                    )
                        
            except Exception as e:
                yield trajectory.trajectory_metadata(
                    title="File Error",
                    content=f"Error loading {file_part.file.name}: {e}"
                )
    
    full_message = user_msg
    if file_content:
        full_message += f"\n\nUploaded file content:{file_content}"
    
    try:
        if not llm or not llm.data:
            raise ValueError("LLM service extension is required but not available")
            
        llm_config = llm.data.llm_fulfillments.get("default")
        
        if not llm_config:
            raise ValueError("LLM service extension provided but no fulfillment available")
        
        yield trajectory.trajectory_metadata(
            title="LLM Ready",
            content=f"Using model: {llm_config.api_model}"
        )
        
        llm_client = OpenAIChatModel(
            model_id=llm_config.api_model,
            base_url=llm_config.api_base,
            api_key=llm_config.api_key,
            parameters=ChatModelParameters(temperature=0.0, stream=True),
            tool_choice_support=set(),
        )
        
        tools = []
        if search_enabled:
            tools.append(DuckDuckGoSearchTool())
        if thinking_mode:
            tools.append(ThinkTool())
        
        requirements = []
        
        if search_enabled:
            requirements.append(
                ConditionalRequirement(
                    DuckDuckGoSearchTool, 
                    max_invocations=2, 
                    consecutive_allowed=False,
                    custom_checks=[lambda state: not is_casual(user_msg)]
                )
            )
        
        if thinking_mode:
            requirements.append(
                ConditionalRequirement(ThinkTool, force_at_step=1, force_after=Tool, consecutive_allowed=False)
            )
        
        base_instructions = """
        You are a helpful AI assistant. Always respond using clean, consistent Markdown.

        ## Text Styling
        - **Bold** for emphasis.
        - `-` for unordered lists; indent nested items by two spaces.
        - `1.` for ordered lists.
        - Keep indentation and spacing readable.

        ## Structure
        - Use `#`, `##`, `###` for headings.
        - Use `---` to separate sections.
        - Use `>` for quotes or summaries.
        - Include links like: [Link Title](https://example.com)

        ## General Style
        - Prioritize readability.
        - Add line breaks between lists and paragraphs for clarity.
        """

        if search_enabled:
            base_instructions += """ For search results, ALWAYS use proper markdown citations: [description](URL).

Examples:
- [OpenAI releases GPT-5](https://example.com/gpt5)
- [AI adoption increases 67%](https://example.com/ai-study)

Use DuckDuckGo for current info, facts, and specific questions. Respond naturally to casual greetings without search."""
        
        base_instructions += """

When files are uploaded, analyze and summarize their content. For data files (CSV/JSON), highlight key insights and patterns."""
        
        style_instructions = {
            "concise": "\n\nIMPORTANT: Keep your responses VERY brief and to the point. Use short sentences. Avoid elaboration unless absolutely necessary. Maximum 2-3 sentences per main point.",
            "standard": "\n\nProvide responses with appropriate detail and context. Include relevant examples when helpful.",
            "detailed": "\n\nIMPORTANT: Provide COMPREHENSIVE and THOROUGH responses. Include extensive explanations, multiple examples, background context, step-by-step breakdowns, and elaborate on all relevant aspects. Be verbose and educational. Use detailed reasoning and provide in-depth analysis."
        }
        
        instructions = base_instructions + style_instructions.get(response_style, "")
        
        tool_names = [tool.__class__.__name__.replace("Tool", "") for tool in tools]
        yield trajectory.trajectory_metadata(
            title="Agent Configured",
            content=f"Tools: {', '.join(tool_names) if tool_names else 'None'} | Requirements: {len(requirements)} active | Streaming: Enabled"
        )
        
        agent = RequirementAgent(
            llm=llm_client, 
            memory=memory,
            tools=tools,
            requirements=requirements,
            instructions=instructions
        )
        
        if is_casual(user_msg):
            yield trajectory.trajectory_metadata(
                title="Processing",
                content="Detected casual message - processing without search"
            )
        else:
            yield trajectory.trajectory_metadata(
                title="Processing",
                content="Analyzing message and determining required actions"
            )
        
        response_text = ""
        search_results = None
        
        def handle_final_answer_stream(data: RequirementAgentFinalAnswerEvent, meta: EventMeta) -> None:
            nonlocal response_text
            if data.delta:
                response_text += data.delta
        
        async for event, meta in agent.run(
            full_message,
            execution=AgentExecutionConfig(max_iterations=20, max_retries_per_step=2, total_max_retries=5),
            expected_output="Markdown format with proper [text](URL) citations for search results." if search_enabled else "Direct response without search citations."
        ).on("final_answer", handle_final_answer_stream):
            
            if meta.name == "final_answer":
                if isinstance(event, RequirementAgentFinalAnswerEvent) and event.delta:
                    yield event.delta
                    continue
            
            if meta.name == "success" and event.state.steps:
                step = event.state.steps[-1]
                if not step.tool:
                    continue
                    
                tool_name = step.tool.name
                
                if tool_name == "final_answer":
                    pass
                elif "search" in tool_name.lower() or "duckduckgo" in tool_name.lower():
                    search_results = getattr(step.output, 'results', None)
                    query = step.input.get("query", "Unknown")
                    count = len(search_results) if search_results else 0
                    
                    yield trajectory.trajectory_metadata(
                        title="Web Search",
                        content=f"Searched for: '{query}' - Found {count} results"
                    )
                elif tool_name == "think":
                    thoughts = step.input.get("thoughts", "Processing...")
                    yield trajectory.trajectory_metadata(
                        title="Thinking",
                        content=thoughts
                    )
                else:
                    yield trajectory.trajectory_metadata(
                        title=f"Tool: {tool_name}",
                        content="Executing specialized tool operation"
                    )
        
        citations, clean_text = extract_citations(response_text, search_results)
        
        if citations:
            yield trajectory.trajectory_metadata(
                title="Citations Processed",
                content=f"Extracted {len(citations)} citation(s) from search results"
            )
            yield citation.citation_metadata(citations=citations)
        
        # Store response in context for session history
        response_message = AgentMessage(text=response_text)
        await context.store(response_message)
        
        yield trajectory.trajectory_metadata(
            title="Complete",
            content="Response delivered successfully"
        )

    except Exception as e:
        yield trajectory.trajectory_metadata(
            title="Error",
            content=f"Error: {e}"
        )
        error_msg = f"Error processing request: {e}"
        yield error_msg
        
        # Store error message too
        await context.store(AgentMessage(text=error_msg))

def run():
    """Start the server with context storage enabled"""
    server.run(
        host=os.getenv("HOST", "127.0.0.1"), 
        port=int(os.getenv("PORT", 8000)),
        context_store=PlatformContextStore()  # Enable session history
    )

if __name__ == "__main__":
    run()