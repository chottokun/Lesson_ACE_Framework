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
会話の履歴と「現在の世界モデル（Current Model）」に基づいて、ユーザーの最新の入力（User Input）を分析してください。

# Inputs
1. **User Input**: "{user_input}"
2. **Current Model**: {current_model}
3. **History**: 
{history_txt}

# Instructions
次の3つの事項を行ってください：
1. **Entity Extraction**: ユーザーの要求に含まれる具体的なエンティティや事実を抽出。
2. **Intent & Problem Class**: 要求に関連する抽象的な問題クラスや意図を特定。
3. **Model Refinement (MFR)**: ユーザーの入力が「現在の世界モデル」に対する変更（制約の追加、訂正、削除）を含んでいるか確認し、差分操作（Diff Ops）を生成してください。
   - 変更がない場合は "stm_diffs": [] としてください。
   - 変更がある場合は以下の操作を使用:
     - `ADD_CONSTRAINT: <内容>` (新しい制約やルールの追加)
     - `MODIFY_ACTION: <内容>` (アクションや手段の変更・修正)
     - `DROP_ENTITY: <名前>` (不要になったエンティティの削除)

# Output JSON Format
{{
    "entities": ["エンティティ1", "エンティティ2"],
    "problem_class": "抽象的な問題クラス",
    "search_query": "検索クエリ文字列",
    "stm_diffs": [
        "ADD_CONSTRAINT: 予算は3000円以内",
        "MODIFY_ACTION: 鍵ではなくカードキーを使用"
    ]
}}
"""

RETRIEVED_CONTEXT_TEMPLATE = "--- 取得されたコンテキスト ---\n{context_str}\n-----------------------"

# --- STM (Short-Term Memory) Templates ---

RESPONSE_STYLE_INSTRUCTIONS = {
    "concise": "簡潔に回答してください。要点のみを述べ、冗長な説明は避けてください。",
    "detailed": "詳細に説明してください。背景情報や関連する考慮事項も含めてください。",
    "evidence-based": "根拠を重視して回答してください。主張には必ず情報源や論拠を明示してください。",
    "step-by-step": "ステップバイステップで説明してください。手順やプロセスを順序立てて示してください。",
    "comparative": "比較・対照の形式で回答してください。選択肢がある場合はメリット・デメリットを明示してください。",
    "tutorial": "チュートリアル形式で回答してください。初心者にもわかるよう、前提知識から丁寧に説明してください。",
    "summary-only": "要約のみで回答してください。結論を1-2文で簡潔に述べてください。"
}

STM_CONTEXT_TEMPLATE = """--- セッション情報 ---
現在時刻: {current_time}
対話ターン: {turn_count}
{style_instruction}
-----------------------
"""
