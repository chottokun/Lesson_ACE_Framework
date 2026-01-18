# 💬 7. APIとプロンプト

ACE Frameworkの内部APIと、動作を制御するプロンプトについて解説します。

## 主要クラスとメソッド

### 1. `ACE_Memory` (`src/ace_rm/memory/core.py`)
長期記憶（SQLite + FAISS）を管理するメインクラスです。

- `add(content, entities, problem_class)`: 新しい知識をSQLiteとFAISSに追加します。
- `add_batch(items)`: 複数の知識を効率的に一括追加します（高速化済み）。
- `search(query, k=3)`: ベクトル検索と全文検索を組み合わせたハイブリッド検索を実行します。
- `update_document(doc_id, content, entities, problem_class)`: 指定したIDのドキュメントを更新し、ベクトルも再計算します。
- `get_all()`: 全ての保存済みドキュメントを取得します。

### 2. `TaskQueue` (`src/ace_rm/memory/queue.py`)
非同期処理用のタスク管理を行います。

- `enqueue_task(user_input, agent_output)`: 対話ペアをキューに追加します。
- `fetch_pending_task()`: 未処理のタスクを1つ取得します。
- `mark_task_complete(task_id)`: タスクを完了状態にします。

### 3. `BackgroundWorker` (`src/ace_rm/workers/background.py`)
バックグラウンド学習を実行するスレッドクラスです。

- `run()`: キューを監視し、タスクがあれば `process_task` を実行するループ。
- `process_task(task)`: LLMを呼び出し、対話の分析と知識の合成（NEW/UPDATE/KEPT）を行います。

## プロンプト管理

プロンプトは `src/ace_rm/prompts/` ディレクトリ配下に、言語ごとに外部化されています。

### 言語の切り替え
環境変数 `ACE_LANG` で制御されます。
- `ACE_LANG=ja`: `ja.py` から読み込み（デフォルト）
- `ACE_LANG=en`: `en.py` から読み込み

### 主要なプロンプト
- `INTENT_ANALYSIS_PROMPT`: **Curator** が使用。ユーザーの意図を分析し、検索クエリやエンティティを抽出します。
- `UNIFIED_ANALYSIS_PROMPT`: **Background Worker** が使用。対話内容と既存知識を比較し、どのように記憶を更新するかを決定します。
- `RETRIEVED_CONTEXT_TEMPLATE`: 検索結果をエージェントのシステムメッセージに整形する際のテンプレート。

**実装の詳細:**
`src/ace_rm/prompts/__init__.py` において、環境変数に基づき動的にモジュールがインポートされます。

---
詳細は [テストと検証](./08-テストと検証.md) を参照してください。
