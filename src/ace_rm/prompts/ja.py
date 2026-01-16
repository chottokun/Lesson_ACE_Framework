# 日本語プロンプトテンプレート


UNIFIED_ANALYSIS_PROMPT = """
このやり取りを分析し、知識ベースに保存または更新すべきか判断してください。
出力は必ず日本語（Japanese）で行ってください。

1. **分析フェーズ**: 
   やり取りから重要な構造的知識（エンティティ、ルール、プロセス）を抽出してください。
   具体的な詳細だけでなく、抽象的な問題クラスも特定してください。

2. **統合判定フェーズ**:
   抽出した知識と「類似する既存の知識」を比較し、アクションを決定してください。

ユーザー: {user_input}
AI: {agent_output}

類似する既存の知識:
{existing_docs}

Output JSON only:
{{
    "should_store": true/false, // 知識として価値があるか
    "action": "NEW" | "UPDATE" | "KEPT", // 保存アクション (should_store=trueの場合のみ有効)
    "target_doc_id": null | <integer_id>, // UPDATEする場合の対象ID
    "analysis": "**具体的なモデル**:\\n[...]\\n\\n**一般化**:\\n[...]", // NEWまたはUPDATE用のコンテンツ
    "entities": ["entity1", "entity2"],
    "problem_class": "problem_class",
    "rationale": "決定の理由"
}}
"""



INTENT_ANALYSIS_PROMPT = """
会話の履歴に基づいて、ユーザーの最新の要求を分析してください。

次の2つの事項を対象としてください：
1. 要求に含まれる具体的なエンティティと事実。
2. 要求に関連する抽象的な問題クラス、構造的パターン、または一般的原則。

ユーザーの要求: "{user_input}"
履歴: 
{history_txt}

Output JSON only:
{{
    "entities": ["エンティティ1", "エンティティ2"],
    "problem_class": "抽象的な問題クラス",
    "search_query": "具体的なエンティティと抽象的な概念を組み合わせた、単一の効果的な検索クエリ文字列"
}}
"""

RETRIEVED_CONTEXT_TEMPLATE = "--- 取得されたコンテキスト ---\n{context_str}\n-----------------------"


LTM_KNOWLEDGE_MODEL_PROMPT = """
以下の情報を分析し、推論に不可欠な最小限の構造モデルを抽出してください。
各項目は箇条書きで、1行30文字以内の「シンボリックな表現」を心がけてください。

---
分析対象:
{context}
---

## エンティティ
永続的なオブジェクトのみ特定。
- 例: User, Session, Config

## 状態変数
変化を追跡すべき動的プロパティ。
- 例: is_authenticated: bool

## アクション
実行可能な操作と「前提条件」→「効果」。
- 例: login(credentials) → session生成

## 制約
破ってはならない不変の境界条件。
- 例: session有効期限 <= 24h

※注意：解決策は書かず、問題の「設計図」のみを出力してください。
"""
