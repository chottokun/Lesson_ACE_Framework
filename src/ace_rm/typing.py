from typing import List, TypedDict, NotRequired
from langchain_core.messages import BaseMessage

class AgentState(TypedDict):
    """
    Represents the state of the ACE agent at any point in the conversation.

    Attributes:
        messages (List[BaseMessage]): The history of messages in the conversation.
        context_docs (List[str]): Documents retrieved by the Curator to provide context.
        extracted_entities (List[str]): Specific entities extracted from the user's input.
        problem_class (str): The abstract problem class identified by the Curator.
        retry_count (int): A counter for handling retries in case of failures.
        lesson_learned (NotRequired[str]): A summary of the lesson learned from the Reflector's analysis.
        should_store (NotRequired[bool]): A flag indicating whether the Reflector decided to store the lesson.
    """
    messages: List[BaseMessage]
    context_docs: List[str]
    extracted_entities: List[str]
    problem_class: str
    retry_count: int
    lesson_learned: NotRequired[str]
    should_store: NotRequired[bool]
