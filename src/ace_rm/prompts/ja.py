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


SYNTHESIZER_PROMPT = """
あなたはAIメモリシステムの「知識合成器」です。
目的は、高品質で重複のない知識ベースを維持することです。
出力（synthesized_content, rationale）は必ず日本語（Japanese）で行ってください。

既存の知識と、最近のやり取りから得られた新しい知識を比較してください。

既存の知識 (ID: {best_match_id}):
{existing_content}

新しい知識:
{new_content}

最善のアクションを決定してください：
1. **UPDATE**: 新しい知識が既存の知識に価値を加え、修正、または洗練させる場合。それらを1つの包括的なエントリにマージします。
2. **KEPT**: 新しい知識が冗長であるか、劣っているか、あるいは既に既存の知識でカバーされている場合。既存の知識をそのまま保持します。
3. **NEW**: 新しい知識が別のエントリとして区別されるべき場合（例：文脈が異なる、矛盾しているが有効な代替案など）。

Output JSON only:
{{
    "action": "UPDATE" | "KEPT" | "NEW",
    "rationale": "決定の簡単な理由",
    "synthesized_content": "マージされたコンテンツ (UPDATEの場合のみ、それ以外はnull)",
    "merged_entities": ["すべてのエンティティの", "リスト"] (UPDATEの場合のみ)
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
