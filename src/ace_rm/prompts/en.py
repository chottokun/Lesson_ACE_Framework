# English Prompt Templates


UNIFIED_ANALYSIS_PROMPT = """
Analyze this interaction and determine if it should be stored or updated in the knowledge base.
Ensure the output is in English.

1. **Analysis Phase**:
   Extract significant structural knowledge (entities, rules, processes) from the interaction.
   Identify not just specific details but also abstract problem classes.

2. **Synthesis Decision Phase**:
   Compare the extracted knowledge with "Similar Existing Knowledge" and decide on an action.

User: {user_input}
AI: {agent_output}

Similar Existing Knowledge:
{existing_docs}

Output JSON only:
{{
    "should_store": true/false, // Is it valuable as knowledge?
    "action": "NEW" | "UPDATE" | "KEPT", // Action (only valid if should_store=true)
    "target_doc_id": null | <integer_id>, // Target ID if UPDATE
    "analysis": "**Specific Model**:\\n[...]\\n\\n**Generalization**:\\n[...]", // Content for NEW or UPDATE
    "entities": ["entity1", "entity2"],
    "problem_class": "problem_class",
    "rationale": "Reason for decision"
}}
"""


SYNTHESIZER_PROMPT = """
You are the "Knowledge Synthesizer" for an AI memory system.
Your goal is to maintain a high-quality, non-redundant knowledge base.

Compare the EXISTING KNOWLEDGE with the NEW KNOWLEDGE derived from a recent interaction.

EXISTING KNOWLEDGE (ID: {best_match_id}):
{existing_content}

NEW KNOWLEDGE:
{new_content}

Determine the best action:
1. **UPDATE**: The NEW knowledge adds value, corrects, or refines the EXISTING knowledge. Merge them into a single, comprehensive entry.
2. **KEPT**: The NEW knowledge is redundant, inferior, or already covered by EXISTING. Keep EXISTING as is.
3. **NEW**: The NEW knowledge is distinct enough to be a separate entry (e.g., different context, contradictory but valid alternative).

Output JSON only:
{{
    "action": "UPDATE" | "KEPT" | "NEW",
    "rationale": "Brief reason for decision",
    "synthesized_content": "Merged content (only for UPDATE, otherwise null)",
    "merged_entities": ["list", "of", "all", "entities"] (only for UPDATE)
}}
"""

INTENT_ANALYSIS_PROMPT = """
Based on the conversation history, analyze the user's latest request.

Target TWO things:
1. Specific entities and facts mentioned in the request.
2. Abstract problem classes, structural patterns, or general principles relevant to the request.

User Request: "{user_input}"
History: 
{history_txt}

Output JSON only:
{{
    "entities": ["entity1", "entity2"],
    "problem_class": "Abstract Problem Class",
    "search_query": "Single effective search query string combining specific entities and abstract concepts"
}}
"""

RETRIEVED_CONTEXT_TEMPLATE = "--- Retrieved Context ---\n{context_str}\n-----------------------"

# --- STM (Short-Term Memory) Templates ---

RESPONSE_STYLE_INSTRUCTIONS = {
    "concise": "Respond concisely. State only the key points and avoid verbose explanations.",
    "detailed": "Provide a detailed explanation. Include background information and related considerations.",
    "evidence-based": "Focus on evidence. Always cite sources or provide supporting arguments for your claims.",
    "step-by-step": "Explain step by step. Present procedures and processes in a logical order.",
    "comparative": "Use a comparative format. When options exist, clearly state the pros and cons of each.",
    "tutorial": "Use a tutorial format. Explain from the basics so that beginners can understand.",
    "summary-only": "Provide a summary only. State the conclusion in 1-2 sentences."
}

STM_CONTEXT_TEMPLATE = """--- Session Info ---
Current time: {current_time}
Dialog turn: {turn_count}
{style_instruction}
-----------------------
"""
