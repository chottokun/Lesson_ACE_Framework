# Agentic Context Engineering Framework Lesson （学習メモ）:

※このレポジトリは勉強中の立場から自分なりに解釈・改変しつつまとめた学習ログです。
This repository documents my learner‑level exploration of the ACE Framework.
It includes personal interpretations, experimental notes, and incremental refinements made while studying the topic.

An LLM agent framework designed to demonstrate persistent memory, structural learning, and adaptive context engineering. ACE goes beyond simple chatbots by actively learning from interactions and retrieving generalized strategies to solve novel problems.

## 🧠 Core Architecture

The ACE Framework operates on a cognitive cycle composed of five key components:

1.  **Curator (Retrieval & Context)**
    *   **Function**: Analyzes user intent and queries the long-term memory.
    *   **Advanced Logic**: Extracts both specific entities (e.g., "5L jug") and abstract problem classes.
    *   **MFR (Model-First Reasoning)**: Compares user input against the current **World Model (STM)** and generates **Diff Operations** (Add/Modify/Drop) to evolve the agent's understanding incrementally.

2.  **Agent (Reasoning & Action)**
    *   **Function**: The core LLM that generates responses or executes tools.
    *   **Context-Aware**: Utilizes the context provided by the Curator to ground its answers in established knowledge or past lessons.

3.  **Reflector (Queuing & Hand-off)**
    *   **Function**: Runs immediately after the agent's response.
    *   **Action**: Enqueues the interaction into a persistent `TaskQueue` (SQLite), ensuring instant feedback.

4.  **Background Worker (Structural Learning)**
    *   **Function**: A dedicated thread that continuously processes the task queue.
    *   **Structural Learning (MFR)**: Deconstructs the conversation into a **Specific Model** and **Generalization**.
    *   **Intelligent Synthesis**: Uses an **LLM-based Synthesizer** (in `BackgroundWorker`) to decide whether to **UPDATE**, **KEEP**, or add as **NEW** knowledge.
    *   **Optimization**: Shared memory/model architecture prevents redundant resource loading.

5.  **Long-Term Memory (LTM)**
    *   **Hybrid Storage**: `ACE_Memory` class combines **SQLite** (documents) and **FAISS** (vectors).
    *   **Task Queue**: `TaskQueue` class manages background jobs independently of the vector store, ensuring **ChromaDB Readiness**.

6.  **Short-Term Memory (STM)**
    *   **Function**: Maintains session-level state that persists across a single conversation.
    *   **User-Configurable**: Response style (concise, detailed, evidence-based, step-by-step, comparative, tutorial, summary-only) can be selected via UI dropdown.
    *   **Function**: Maintains session-level state that persists across a single conversation.
    *   **World Model**: Holds a structured representation of the current task (`Constraints`, `Actions`, `Entities`). This model persists across turns and evolves via MFR updates.
    *   **User-Configurable**: Response style (concise, detailed, etc.) can be selected via UI dropdown.
    *   **Context Injection**: Current time, turn count, and style instructions are injected as a `SystemMessage`.

## ⚙️ 処理フローの可視化 (Visualization)

このシステムのアーキテクチャは、**同期的な対話ループ**と**非同期的な学習ループ**という2つの主要なサイクルで構成されています。これにより、ユーザーへの即時応答性と、バックグラウンドでの継続的な自己改善を両立させています。

```mermaid
graph TD
    %% サブグラフ定義の修正: subgraph ID ["表示名"] の形式に変更
    subgraph UI ["User Interface (app.py)"]
        User["👤 ユーザー"] -->|"1. メッセージ入力"| Gradio["🌐 Gradio UI"]
    end

    subgraph ACE ["ACE Agent - 同期処理 (ace_framework.py)"]
        Gradio -->|"2. ace_app.invoke() 呼び出し"| Curator
        Curator["🧠 Curator <br> 意図分析・文脈検索"] -->|"4. 関連情報を検索"| Memory
        Memory -->|"5. 検索結果を返す"| Curator
        Curator -->|"6. 文脈を注入"| Agent
        Agent["🤖 Agent <br> 応答生成"] -->|"7. 対話内容を渡す"| Reflector
        Reflector["📝 Reflector <br> 対話をキューに追加"] -->|"8. タスクをキューに保存"| TaskQueue
    end

    subgraph LTM ["Long-Term Memory (ace_framework.py)"]
        Memory["📚 長期記憶 <br> (SQLite + FAISS)"]
        TaskQueue["📦 タスクキュー <br> (SQLite)"]
    end

    subgraph BG ["Background Learning - 非同期処理 (ace_framework.py)"]
        BG_Worker["⚙️ Background Worker <br> 定期的にキューを監視"] -->|"11. 未処理タスクを取得"| TaskQueue
        TaskQueue -->|"12. タスクを渡す"| BG_Worker
        BG_Worker -->|"13. 対話を分析・一般化 (LLM)"| BG_Worker
        BG_Worker -->|"14. 学習した知識を保存"| Memory
    end

    %% Final Output to User
    Reflector -->|"9. 応答を返す"| Gradio
    Gradio -->|"10. 応答を表示"| User

    %% Style definitions
    style User fill:#c9f,stroke:#333,stroke-width:2px
    style Gradio fill:#ccf,stroke:#333,stroke-width:2px
    style BG_Worker fill:#f96,stroke:#333,stroke-width:2px
    style Memory fill:#9cf,stroke:#333,stroke-width:2px
    style TaskQueue fill:#9cf,stroke:#333,stroke-width:2px
```

### Part 1: 同期的な対話ループ (ユーザーへの即時応答)

このループは、ユーザーがメッセージを送信してからエージェントが応答を返すまでの、リアルタイムで行われる処理です。

-   **ユーザー入力**
    -   ユーザーがGradioのUIにメッセージを入力し、「送信」ボタンを押します。
    -   **関連コード**: `app.py` - `gr.Textbox` / `gr.Button`

-   **エージェント呼び出し**
    -   Gradioのイベントが`app.py`の`process_chat`関数をトリガーします。この関数は、対話履歴をLangChainのメッセージ形式に変換し、`ace_app.invoke()`を呼び出してACE Agentのワークフローを開始します。
    -   **関連コード**: `app.py` - `process_chat`関数

-   **Curator: 意図分析と文脈検索**
    -   ワークフローの最初のノードである`curator_node`が実行されます。LLMを呼び出してユーザーの最新のメッセージから「具体的なエンティティ」と「抽象的な問題クラス」を抽出し、それに基づいて長期記憶を検索するためのクエリを生成します。
    -   **関連コード**: `ace_framework.py` - `curator_node`関数

-   **長期記憶からの検索**
    -   `curator_node`は`ACE_Memory`クラスの`search`メソッドを呼び出します。このメソッドは、FAISSによるベクトル検索とSQLite FTS5によるキーワード検索を組み合わせたハイブリッド検索を実行し、関連する過去の知識（ドキュメント）を取得します。
    -   **関連コード**: `ace_framework.py` - `ACE_Memory.search`メソッド

-   **Agent: 応答生成**
    -   Curatorによって検索された知識は、システムメッセージとして対話履歴の先頭に注入（コンテキストとして追加）されます。この強化されたコンテキストを受け取った`agent_node`が、LLMを呼び出してユーザーへの最終的な応答を生成します。
    -   **関連コード**: `ace_framework.py` - `agent_node`関数

-   **Reflector: 対話の記録**
    -   Agentの応答後、`reflector_node`が実行されます。このノードの役割は、今回の対話（ユーザー入力とAgent応答のペア）を分析・学習させるために、`ACE_Memory`の`enqueue_task`メソッドを呼び出してタスクキュー（SQLiteテーブル）に保存することです。この処理は非常に軽量で、ユーザーを待たせることはありません。
    -   **関連コード**: `ace_framework.py` - `reflector_node`関数, `ACE_Memory.enqueue_task`メソッド

-   **ユーザーへの応答**
    -   `reflector_node`の処理が終わると、同期処理である`ace_app.invoke()`が完了します。`app.py`の`process_chat`関数は最終的な応答テキストをGradioのチャットボットに返し、ユーザーの画面に応答が表示されます。
    -   **関連コード**: `app.py` - `process_chat`関数

### Part 2: 非同期的な学習ループ (バックグラウンドでの自己改善)

このループは、メインの対話スレッドとは独立してバックグラウンドで実行され、エージェントが過去の対話から学習し、長期記憶を豊かにしていくプロセスを担います。

-   **タスクの取得**
    -   `app.py`の起動と同時に開始された`BackgroundWorker`スレッドが、定期的にタスクキューをチェックします。`ACE_Memory.fetch_pending_task`メソッドを使い、ステータスが'pending'の最も古いタスクを1件取得します。
    -   **関連コード**: `ace_framework.py` - `BackgroundWorker.run`, `ACE_Memory.fetch_pending_task`

-   **分析と一般化**
    -   取得したタスク（対話ペア）を`BackgroundWorker.process_task`メソッドで処理します。ここでは、「具体的なモデルの分析」と「抽象的なパターンの一般化」を行うように設計されたプロンプトを使ってLLMを再度呼び出します。このステップで、対話から再利用可能な知識（教訓や戦略）が抽出されます。
    -   **関連コード**: `ace_framework.py` - `BackgroundWorker.process_task`

-   **長期記憶への保存**
    -   LLMによる分析結果に、保存すべき価値がある（`should_store: true`）と判断された場合、抽出された知識が`ACE_Memory.add`メソッドを通じて長期記憶に保存されます。具体的には、テキストデータはSQLiteに、そのベクトル表現はFAISSインデックスに追加されます。これにより、未来の対話でCuratorがこの新しい知識を検索・利用できるようになります。
    -   **関連コード**: `ace_framework.py` - `ACE_Memory.add`メソッド

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

Create a `.env` file in the root directory (see `.env.example` for a template). All configuration is centrally managed in `src/ace_rm/config.py`.

```env
# --- LLM Settings ---
LLM_MODEL=gpt-oss-120b
LLM_API_KEY=your_api_key_here
LLM_BASE_URL=https://api.ai.sakura.ad.jp/v1/
LLM_TEMPERATURE=0.0

# --- Multi-language Settings ---
# "ja" for Japanese, "en" for English
ACE_LANG=ja

# --- Embedding Settings ---
# cl-nagoya/ruri-v3-30m is optimized for Japanese retrieval
ACE_EMBEDDING_MODEL=cl-nagoya/ruri-v3-30m
ACE_DISTANCE_METRIC=cosine
ACE_DISTANCE_THRESHOLD=0.7
ACE_DEVICE=cpu # Optional: cpu or cuda (default: auto-detection)

# --- Multi-user Settings ---
# "shared" (Default): All users interact with a single, global memory.
# "isolated": Each user session gets a private, independent memory.
LTM_MODE=shared
```

## ⚡ Optimization & Scalability

The framework has been refactored for high-performance and future-proof scalability:

-   **Batch Insertion (30x Speedup)**: Implemented `add_batch` logic. Bulk memory operations are now processed in a single transaction/FAISS update, reducing insertion time from 31ms to ~1ms per document.
-   **Resource Sharing**: Optimized `BackgroundWorker` to share a single `SentenceTransformer` instance with the Agent. This results in **50% less RAM usage** during concurrent operation.
-   **Vector DB Hardware Acceleration**: FAISS automatically utilizes GPU (CUDA) if available for indexing. The embedding model device can be explicitly set via `ACE_DEVICE` in `.env`.
-   **ChromaDB Readiness**: The modular split between `ACE_Memory` and `TaskQueue` allows swapping the vector backend to ChromaDB without affecting the background processing logic.

## 🖥️ Usage

### Interactive Web UI (Gradio)

The main entry point is a 3-pane Gradio interface that visualizes the agent's internal thought process.

```bash
uv run python src/ace_rm/app.py
```

*   **Left Pane**: Chat interface.
*   **Center Pane**: Debug view showing **Curator** retrieval, **STM (World Model)** state, **LTM Status**, and **Background Processing** status.
*   **Right Pane**: Live view of the Long-Term Memory database.

#### 🎛️ Response Style (STM Settings)

In the debug panel, you can select a response style to control how the agent formats its answers:

| Style | Description |
|:---|:---|
| **簡潔 (Concise)** | Brief, key points only |
| **詳細 (Detailed)** | Comprehensive with background info |
| **根拠重視 (Evidence-based)** | Cites sources and reasoning |
| **ステップバイステップ (Step-by-step)** | Ordered procedural explanation |
| **比較・対照 (Comparative)** | Pros/cons analysis |
| **チュートリアル (Tutorial)** | Beginner-friendly walkthrough |
| **要約のみ (Summary-only)** | 1-2 sentence conclusion |

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
│   ├── memory/           # Memory storage & Vector search (ACE_Memory, TaskQueue)
│   ├── agent/            # LangGraph agent definitions & nodes
│   ├── workers/          # Background processing (BackgroundWorker)
│   ├── prompts/          # Externalized system prompts (multilingual)
│   ├── utils/            # Shared utilities (Embedding model manager)
│   ├── config.py         # Centralized configuration management
│   ├── ace_framework.py  # Facade module (Unified interface for compatibility)
│   └── app.py            # Gradio UI application
├── tests/
│   ├── manual_test_memory_flow.py 
│   ├── benchmark_ace.py  # Performance measurement tool
│   └── ...
├── user_data/            # Isolated session data (SQLite/FAISS)
└── pyproject.toml
```

## References
- Agentic Context Engineering: Evolving Contexts for Self-Improving Language Models. arXiv: [2510.04618](https://arxiv.org/abs/2510.04618)
- Model-First Reasoning LLM Agents: Reducing Hallucinations through Explicit Problem Modeling. arXiv: [2512.14474](https://arxiv.org/abs/2512.14474)
