from tenacity import retry, stop_after_attempt, wait_exponential

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=5))
def call_llm_with_retry(llm, messages):
    """
    Invokes the language model with a retry mechanism.

    This function calls the llm.invoke method with the provided messages and
    automatically retries on failure with exponential backoff.

    Args:
        llm: The language model instance.
        messages: The list of messages to send to the language model.

    Returns:
        The response from the language model.
    """
    return llm.invoke(messages)
