# ChromaDB統合・移行詳細実装計画書

## 1. 目的
`plan_rag.md` で提案された「案A（統合検索）」および「案D（全面移行）」を実現するための具体的なエンジニアリング工程を定義します。本計画では、将来的にバックエンドを容易に切り替え可能にするためのリファクタリングを優先します。

## 2. フェーズ1：リトリーバー・インターフェースの抽象化

### 2.1. BaseRetriever クラスの定義
`src/ace_rm/memory/base.py` を新設し、抽象基底クラスを定義します。

```python
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

class BaseRetriever(ABC):
    @abstractmethod
    def search(self, query: str, k: int = 3, **kwargs) -> List[str]:
        pass

    @abstractmethod
    def add(self, content: str, entities: List[str] = [], problem_class: str = ""):
        pass

    @abstractmethod
    def get_all(self) -> List[Dict[str, Any]]:
        pass

    @abstractmethod
    def clear(self):
        pass
```

### 2.2. 現行 ACE_Memory のリファクタリング
`src/ace_rm/memory/core.py` の `ACE_Memory` を `BaseRetriever` を継承するように変更します。また、`find_similar_vectors` などの内部依存メソッドもインターフェースに組み込みます。

## 3. フェーズ2：ChromaDB リトリーバーの実装

### 3.1. ChromaRetriever の実装
`src/ace_rm/memory/chroma.py` を作成します。

- **接続管理**: `chromadb.PersistentClient` を使用し、ローカルディレクトリにデータを保存。
- **メタデータ設計**:
    - `entities`: JSON文字列として保存。
    - `problem_class`: 文字列として保存。
    - `source`: "internal" または "external" のフラグ。
- **検索ロジック**: `collection.query` を使用。`where` 句を用いて `problem_class` によるフィルタリングにも対応させます。

## 4. フェーズ3：マルチソース・プロキシの実装

### 4.1. ACE_MemoryOrchestrator の導入
複数のリトリーバーを束ねるクラスを作成し、`Curator` からは単一の窓口として見えるようにします。

```python
class ACE_MemoryOrchestrator(BaseRetriever):
    def __init__(self, primary: BaseRetriever, secondaries: List[BaseRetriever]):
        self.primary = primary
        self.secondaries = secondaries

    def search(self, query: str, k: int = 3, **kwargs) -> List[str]:
        # 各リトリーバーから検索し、重複排除して返す
        results = self.primary.search(query, k, **kwargs)
        for s in self.secondaries:
            results.extend(s.search(query, k, **kwargs))
        return list(dict.fromkeys(results))[:k]
```

## 5. フェーズ4：全面移行（案D）の実行ステップ
SQLite + FAISS から ChromaDB へ完全に移行する場合の手順です。

1. **データ移行スクリプトの作成**:
   現行の `ace_memory.db` から全件抽出し、`ChromaRetriever.add_batch` で一括投入するユーティリティを作成。
2. **依存関係の置換**:
   `src/ace_rm/ace_framework.py` でエクスポートしている `ACE_Memory` を `ChromaRetriever`（またはそれを継承した新 `ACE_Memory`）に差し替え。
3. **TaskQueue の分離維持**:
   `TaskQueue` は SQLite のままで維持し、`ACE_Memory` との DB ファイル共有を解消（独立した `ace_tasks.db` 等へ）。

## 6. フェーズ5：UI・設定の更新

- **.env の拡張**:
  - `ACE_LTM_BACKEND=sqlite_faiss` | `chromadb`
  - `CHROMA_DB_PATH=./chroma_db`
- **app.py の更新**:
  - デバッグパネルに「検索ソース」の統計情報を表示。
  - バックエンドの状態確認ランプを追加。

## 7. テスト計画
- **Unit Test**: `BaseRetriever` の各実装が同じ入出力仕様を満たすか確認。
- **Integration Test**: `tests/manual_test_memory_flow.py` を実行し、バックエンドを ChromaDB に切り替えても「Jug Puzzle」の学習と検索が正常に行われるか検証。
- **Benchmark**: 大量データ投入時の検索レイテンシを FAISS と比較。

## 8. 結論
本計画により、ACE Framework は特定のベクトルDB実装に依存しない柔軟な構造へと進化します。まずは抽象化レイヤーの導入（フェーズ1）から着手し、並行運用（フェーズ2, 3）を経て、最終的に要件に応じたバックエンドの選択（フェーズ4）を可能にします。
