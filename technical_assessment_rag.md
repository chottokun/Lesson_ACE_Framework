# RAG基盤の統合・移行に関する技術アセスメント報告書

## 1. 概要
本ドキュメントは、ACE Frameworkの既存LTM基盤（SQLite + FAISS）をChromaDBへ移行、または統合検索として活用する場合の技術的妥当性と具体的なハードルを、コードレベルで評価したものです。

## 2. 現行実装のボトルネック解析

### 2.1. ID管理の依存性 (Integer vs. String)
- **現状**: `src/ace_rm/memory/core.py` は、SQLiteの `INTEGER PRIMARY KEY` を `doc_id` として使用しています。
- **課題**: ChromaDBのIDは原則として文字列（String）です。
- **影響範囲**: `BackgroundWorker.process_task` が `find_similar_vectors` から返されたIDを使って `get_document_by_id` を呼び出す際、型の不一致が発生します。
- **対策**: `BaseRetriever` インターフェースではIDをすべて `str` として扱い、SQLite実装側で型変換（`str(id)` ⇄ `int(id)`）を行うブリッジ層が必要です。

### 2.2. ハイブリッド検索の再現性
- **現状**: `ACE_Memory.search` は、FAISSによるベクトル検索でヒットが `k` 件に満たない場合、SQLite FTS5（全文検索）をフォールバックとして使用します。
- **課題**: ChromaDB標準では、BM25等のキーワード検索とベクトル検索の「フォールバック」や「リランキング」の挙動がSQLite FTS5とは異なります。
- **評価**: SQLite FTS5は日本語のトークナイズ（MeCab等）なしでも `icu` 等で高度な検索が可能ですが、ChromaDB単体で同等の検索品質を維持するには、適切なトークナイザーの設定が必要です。

### 2.3. 非同期整合性とトリガー
- **現状**: SQLite FTS5テーブルの更新は、SQLの `TRIGGER` によって自動化されています。
- **課題**: ChromaDB移行後は、アプリケーション層（Python）でドキュメント追加とベクトルインデックス更新の整合性を担保する必要があります。
- **評価**: `FileLock` によるFAISSファイルの排他制御が不要になるため、並行書き込み時の堅牢性はChromaDBの方が高くなります。

## 3. 統合パターンの比較評価

| 評価項目 | 案A: 統合検索 (Multi-Source) | 案D: 全面移行 (Replacement) |
| :--- | :--- | :--- |
| **開発コスト** | 低（既存コードを維持） | 中（リファクタリング大） |
| **性能** | 劣化の可能性（並列検索のオーバヘッド） | 向上（単一DBへのアクセス） |
| **メンテナンス性** | 複雑（2つの基盤を管理） | 高（基盤が1つに集約） |
| **セッション分離** | 容易（既存のファイル分離を維持） | 考慮が必要（Collection名での分離） |

## 4. PoC（概念実証）用インターフェース案

移行を確実にするため、以下の `BaseRetriever` 定義に基づき、現行の `ACE_Memory` をリラップすることを推奨します。

```python
# src/ace_rm/memory/base.py (新規)
class BaseRetriever(ABC):
    @abstractmethod
    def search(self, query: str, k: int = 3) -> List[str]:
        """検索（文字列のリストを返す）"""
        pass

    @abstractmethod
    def find_similar(self, content: str, threshold: float) -> List[Tuple[str, float]]:
        """ベクトル近傍検索（IDとスコアのリストを返す）"""
        pass

    @abstractmethod
    def get_by_id(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """ID指定取得"""
        pass
```

## 5. 結論と提言
コード解析の結果、**「案D：全面移行」は技術的に十分可能ですが、ID型の抽象化という破壊的変更を伴う**ことが判明しました。

**推奨ステップ**:
1.  **フェーズ0**: IDを `str` に統一するリファクタリングのみを先行実施。
2.  **フェーズ1**: `BaseRetriever` 経由で既存の SQLite + FAISS を呼び出す。
3.  **フェーズ2**: ChromaDB実装を追加し、`.env` で切り替え可能にする。

これにより、既存の `BackgroundWorker` や `Curator` のロジックを壊すことなく、安全に基盤を刷新できると評価します。
