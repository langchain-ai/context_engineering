# 🧱 Context Engineering with LangGraph 

Agents need context (e.g., instructions, external knowledge, tool feedback) to perform tasks. Context engineering is the art and science of filling the context window with just the right information at each step of an agent’s trajectory. This repository has a set of notebooks in the `context_engineering` folder that cover different strategies for context engineering, including **write, select, compress, and isolate**. For each, we explain how LangGraph is designed to support it with examples. 

<img width="1231" height="448" alt="Screenshot 2025-07-13 at 2 57 28 PM" src="https://github.com/user-attachments/assets/8e7b59e0-4bb0-48f6-aeba-2d789ada55e3" />

## 🚀 Quickstart 

### Prerequisites
- Python 3.9 or higher
- [uv](https://docs.astral.sh/uv/) package manager
- [Deno](https://docs.deno.com/runtime/getting_started/installation/) required only for the optional legacy sandbox example in the `4_isolate_context.ipynb` notebook; `langchain-sandbox` is no longer part of the default install because it is incompatible with the LangChain 1.x dependency set

### Installation
1. Clone the repository and activate a virtual environment:
```bash
git clone https://github.com/langchain-ai/context_engineering
cd context_engineering
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

2. Install dependencies:
```bash
uv pip install -r requirements.txt
```

3. Set up environment variables for the model provider(s) you want to use:
```bash
export OPENAI_API_KEY="your-openai-api-key"
export ANTHROPIC_API_KEY="your-anthropic-api-key"
```

4. You can then run the notebooks in the `context_engineering` folder:

```
context_engineering/
├── 1_write_context.ipynb      # Examples of saving context externally
├── 2_select_context.ipynb     # Examples of retrieving relevant context
├── 3_compress_context.ipynb   # Examples of context compression techniques
└── 4_isolate_context.ipynb    # Examples of context isolation methods
```

## 📚 Background 

As Andrej Karpathy puts it, LLMs are like a [new kind of operating system](https://www.youtube.com/watch?si=-aKY-x57ILAmWTdw&t=620&v=LCEmiRjPEtQ&feature=youtu.be). The LLM is like the CPU and its [context window](https://docs.anthropic.com/en/docs/build-with-claude/context-windows) is like the RAM, serving as the model’s working memory. Just like RAM, the LLM context window has limited [capacity](https://lilianweng.github.io/posts/2023-06-23-agent/) to handle various sources of context. And just as an operating system curates what fits into a CPU’s RAM, we can think about “context engineering” playing a similar role. [Karpathy summarizes this well](https://x.com/karpathy/status/1937902205765607626):

> [Context engineering is the] ”…delicate art and science of filling the context window with just the right information for the next step.”

What are the types of context that we need to manage when building LLM applications? We can think of context engineering as an [umbrella](https://x.com/dexhorthy/status/1933283008863482067) that applies across a few different context types:

- **Instructions** – prompts, memories, few‑shot examples, tool descriptions, etc
- **Knowledge** – facts, memories, etc
- **Tools** – feedback from tool calls

## Agent Challenges

However, long-running tasks and accumulating feedback from tool calls mean that agents often utilize a large number of tokens. This can cause numerous problems: it can [exceed the size of the context window](https://cognition.ai/blog/kevin-32b), balloon cost / latency, or degrade agent performance. Drew Breunig [nicely outlined](https://www.dbreunig.com/2025/06/22/how-contexts-fail-and-how-to-fix-them.html) a number of specific ways that longer context can cause perform problems. 

With this in mind, [Cognition](https://cognition.ai/blog/dont-build-multi-agents) called out the importance of context engineering with agents:

> “Context engineering” … is effectively the #1 job of engineers building AI agents.

[Anthropic](https://www.anthropic.com/engineering/built-multi-agent-research-system) also laid it out clearly:

> *Agents often engage in conversations spanning hundreds of turns, requiring careful context management strategies.*
>

## Context Engineering Strategies

In this repo, we cover some common strategies — write, select, compress, and isolate — for agent context engineering by reviewing various popular agents and papers. We then explain how LangGraph is designed to support them!

* **Writing context** - saving it outside the context window to help an agent perform a task.
* **Selecting context** - pulling it into the context window to help an agent perform a task.
* **Compressing context** - retaining only the tokens required to perform a task.
* **Isolating context** - splitting it up to help an agent perform a task.

### 1. Write Context
**Description**: Saving information outside the context window to help an agent perform a task.

### 📚 **What's Covered in [1_write_context.ipynb](context_engineering/1_write_context.ipynb)**
- **Scratchpads in LangGraph**: Using state objects to persist information during agent sessions
  - StateGraph implementation with TypedDict for structured data
  - Writing context to state and accessing it across nodes
  - Checkpointing for fault tolerance and pause/resume workflows
- **Memory Systems**: Long-term persistence across multiple sessions
  - InMemoryStore for storing memories with namespaces
  - Integration with checkpointing for comprehensive memory management
  - Examples of storing and retrieving jokes with user context

## 2. Select Context
**Description**: Pulling information into the context window to help an agent perform a task.

### 📚 **What's Covered in [2_select_context.ipynb](context_engineering/2_select_context.ipynb)**
- **Scratchpad Selection**: Fetching specific context from agent state
  - Selective state access in LangGraph nodes
  - Multi-step workflows with context passing between nodes
- **Memory Retrieval**: Selecting relevant memories for current tasks
  - Namespace-based memory retrieval
  - Context-aware memory selection to avoid irrelevant information
- **Tool Selection**: RAG-based tool retrieval for large tool sets
  - LangGraph Bigtool library for semantic tool search
  - Embedding-based tool description matching
  - Examples with math library functions and semantic retrieval
- **Knowledge Retrieval**: RAG implementation for external knowledge
  - Vector store creation with document splitting
  - Retriever tools integrated with LangGraph agents
  - Multi-turn conversations with context-aware retrieval

## 3. Compress Context
**Description**: Retaining only the tokens required to perform a task.

### 📚 **What's Covered in [3_compress_context.ipynb](context_engineering/3_compress_context.ipynb)**
- **Conversation Summarization**: Managing long agent trajectories
  - End-to-end conversation summarization after task completion
  - Token usage optimization (demonstrated reduction from 115k to 60k tokens)
- **Tool Output Compression**: Reducing token-heavy tool responses
  - Summarization of RAG retrieval results
  - Integration with LangGraph tool nodes
  - Practical examples with blog post retrieval and summarization
- **State-based Compression**: Using LangGraph state for context management
  - Custom state schemas with summary fields
  - Conditional summarization based on context length

## 4. Isolate Context
**Description**: Splitting up context to help an agent perform a task.

### 📚 **What's Covered in [4_isolate_context.ipynb](context_engineering/4_isolate_context.ipynb)**
- **Multi-Agent Systems**: Separating concerns across specialized agents
  - Supervisor architecture for task delegation
  - Specialized agents with isolated context windows (math expert, research expert)
  - LangGraph Supervisor library implementation
- **Sandboxed Environments**: Isolating context in execution environments
  - PyodideSandboxTool for secure code execution
  - State isolation outside the LLM context window
  - Examples of context storage in sandbox variables
- **State-based Isolation**: Using LangGraph state schemas for context separation
  - Structured state design for selective context exposure
  - Field-based isolation within agent state objects

