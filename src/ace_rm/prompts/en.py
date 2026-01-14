# English Prompt Templates

ANALYSIS_PROMPT = """
Analyze this interaction. Extract Structural Knowledge (MFR) and General Principles.

1. **Specific Analysis**: If a problem is presented, define:
   - Entities, State Variables, Actions, Constraints.
   - Summary of the solution.

2. **Abstraction & Generalization**:
   - Abstract the specific details into a general pattern or rule.
   - Identify the underlying problem class (e.g., "Constraint Satisfaction", "Resource Allocation").
   - Define a general strategy derived from this instance.

User: {user_input}
AI: {agent_output}

Output JSON only:
{{
    "analysis": "**Specific Model**:\\n[Details...]\\n\\n**Generalization**:\\n[Details...]",
    "entities": ["list", "of", "key", "entities"],
    "problem_class": "Identified Problem Class",
    "should_store": true/false
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
