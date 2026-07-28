from langchain.agents.middleware import wrap_model_call
from langchain_core.messages import trim_messages

def make_trim_middleware(max_tokens: int):
    @wrap_model_call(name="TrimMessagesMiddleware")
    def trim(request, handler):
        trimmed = trim_messages(
            request.messages,
            strategy="last",
            token_counter=request.model,
            max_tokens=max_tokens,
            start_on="human",
        )
        return handler(request.override(messages=trimmed))
    return trim
