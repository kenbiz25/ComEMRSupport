
from openai import OpenAI
from config.settings import settings

_client = None

def get_openai():
    global _client
    if _client is None:
        _client = OpenAI(api_key=settings.OPENAI_API_KEY)
    return _client

def chat_complete(prompt: str) -> str:
    client = get_openai()
    resp = client.chat.completions.create(
        model=settings.OPENAI_MODEL,
        messages=[
            {"role": "system", "content": "Be accurate, concise, and policy-compliant."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )
    return resp.choices[0].message.content.strip()