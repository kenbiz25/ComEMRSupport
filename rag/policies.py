
def allow_or_block(user_query: str) -> bool:
    blocked = ["hack", "bypass security", "crack", "illegal"]
    return not any(x in user_query.lower() for x in blocked)

def redact_sensitive(text: str) -> str:
    for token in ["api_key"]:
        text = text.replace(token, "[redacted]")
    return text
