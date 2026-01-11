# Agentic Context Engineering Framework Lesson （学習メモ）:

※このレポジトリは勉強中の立場から自分なりに解釈・改変しつつまとめた学習ログです。
This repository documents my learner‑level exploration of the ACE Framework.
It includes personal interpretations, experimental notes, and incremental refinements made while studying the topic.

An LLM agent framework designed to demonstrate persistent memory, structural learning, and adaptive context engineering. ACE goes beyond simple chatbots by actively learning from interactions and retrieving generalized strategies to solve novel problems.

## 🧠 Core Architecture

The ACE Framework operates on a cognitive cycle composed of five key components:

1.  **Curator (Retrieval & Context)**
    *   **Function**: Analyzes user intent and queries the long-term memory.
    *   **Advanced Logic**: Extracts both specific entities (e.g., "5L jug") and abstract problem classes (e.g., "Constraint Satisfaction"). It injects relevant past experiences into the prompt context *before* the agent generates a response.

2.  **Agent (Reasoning & Action)**
    *   **Function**: The core LLM that generates responses or executes tools.
    *   **Context-Aware**: Utilizes the context provided by the Curator to ground its answers in established knowledge or past lessons.

3.  **Reflector (Queuing & Hand-off)**
    *   **Function**: Runs immediately after the agent's response.
    *   **Action**: Instead of blocking the user for analysis, it **queues** the interaction into a persistent task queue, ensuring instant feedback to the user.

4.  **Background Worker (Async Analysis)**
    *   **Function**: A dedicated thread that continuously processes the task queue.
    *   **Structural Learning (MFR)**: Performs the heavy lifting of deconstructing the conversation into a **Specific Model** and **Generalization**.
    *   **Benefits**: Achieves **High Responsiveness** (UI doesn't freeze) and **Zero Data Loss** (tasks are persisted in DB until successfully processed).

5.  **Long-Term Memory**
    *   **Hybrid Storage**: Combines **SQLite** for structured metadata/text and **FAISS** for vector embedding search.
    *   **Task Queue**: Uses a persistent SQLite table to manage background jobs, ensuring no insights are lost even across restarts.
    *   **Persistence**: Knowledge survives application restarts, allowing the agent to "grow" over time.

## 🚀 Setup & Installation

This project uses `uv` for fast and reliable dependency management.

### Prerequisites
*   Python 3.10+
*   `uv` installed (`curl -LsSf https://astral.sh/uv/install.sh | sh`)

### Installation

```bash
# 1. Clone the repository
git clone <repository_url>
cd ace_rm

# 2. Install dependencies
uv sync
```

### Environment Configuration

Create a `.env` file in the root directory:

```env
# Required: Compatible OpenAI API Key (e.g., Sakura, OpenAI, Azure)
SAKURA_API_KEY=your_api_key_here
```

## 🖥️ Usage

### Interactive Web UI (Gradio)

The main entry point is a 3-pane Gradio interface that visualizes the agent's internal thought process.

```bash
uv run python src/ace_rm/app.py
```

*   **Left Pane**: Chat interface.
*   **Center Pane**: Debug view showing **Curator** retrieval and **Reflector** analysis in real-time.
*   **Right Pane**: Live view of the Long-Term Memory database.

### Command Line Interface

You can also interact with the core logic via provided test scripts.

## 🧪 Testing & Verification

### Manual Memory Flow Test

We provide a specialized script to verify the agent's cognitive loop (Learn -> Retrieve -> Transfer). This script simulates a "Water Jug Puzzle" scenario to demonstrate structural learning.

```bash
uv run python tests/manual_test_memory_flow.py
```

**What this script does:**

1.  **Step 1 (Learning)**:
    *   Sends a query: *"How to measure 4L using 3L and 5L jugs?"*
    *   **Expectation**: The Agent solves it. The **Reflector** analyzes the solution, abstracts it into a "Water Jug / Diophantine Reachability" strategy, and stores it in memory.

2.  **Step 2 (Transfer)**:
    *   Sends a follow-up query: *"Can you apply the same strategy to 5L and 8L jugs to measure 2L?"*
    *   **Expectation**: The **Curator** retrieves the generalized strategy learned in Step 1. The Agent applies this strategy to the new variables (5L & 8L) to solve the new problem without starting from scratch.

**Output Interpretation:**
*   Look for `[Reflector] Should Store: True` in Step 1.
*   Look for `[Curator] Found knowledge about: ...` in Step 2.

### Unit Tests

Run the standard test suite to verify individual components.

```bash
uv run pytest
```

## 📂 Project Structure

```text
ace_rm/
├── src/ace_rm/
│   ├── ace_framework.py  # Core logic (Graph definition, Nodes, Memory)
│   └── app.py            # Gradio UI application
├── tests/
│   ├── manual_test_memory_flow.py # End-to-end cognitive flow verification
│   └── ...
├── docs/                 # Architecture and planning documents
├── ace_memory.db         # SQLite database (auto-generated)
├── ace_memory.faiss      # Vector index (auto-generated)
└── pyproject.toml        # Project configuration
```

## References
- Agentic Context Engineering: Evolving Contexts for Self-Improving Language Models. arXiv: [2510.04618](https://arxiv.org/abs/2510.04618)
- Model-First Reasoning LLM Agents: Reducing Hallucinations through Explicit Problem Modeling. arXiv: [2512.14474](https://arxiv.org/abs/2512.14474)
