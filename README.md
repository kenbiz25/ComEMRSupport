
# ComEMR WhatsApp Support Bot

AI Powered WhatsApp chatbot for **ComEMR Support** built using **FastAPI**, **WhatsApp Cloud API**, and **OpenAI**.  
It uses Retrieval-Augmented Generation (RAG) for context-aware answers and sanitizes responses to avoid leaking internal KB details.

---

## ✅ Features
- Responds to user queries via WhatsApp
- Uses RAG for intelligent, context-based answers
- Sanitizes responses (no KB paths or internal details)
- Supports sending **text**, **media**, and **templates**
- Configurable guardrails for security and compliance

---

## ✅ Project Structure
```
apps/
  whatsapp_gateway/   # WhatsApp routes and send logic
config/              # Settings and rate limit configs
core/                # Indexing, KB, FAISS store
jobs/                # Background tasks (e.g., reindex KB)
rag/                 # Composer and retrieval logic
tests/               # Unit tests
prompts/             # System prompt templates
app.py               # FastAPI entry point
requirements.txt     # Dependencies
```

---

## ✅ Setup Instructions

### 1. Clone the Repo
```bash
git clone https://github.com/kenbiz25/ComEMRSupport
cd ComEMRSupport
```

### 2. Create Virtual Environment
```bash
python -m venv venv
# Activate:
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables
Create a `.env` file in the root:
```
WHATSAPP_PHONE_ID=<your-phone-id>
META_WHATSAPP_TOKEN=<your-meta-token>
META_VERIFY_TOKEN=ComEMR1234
OPENAI_API_KEY=<your-openai-key>
KB_DIR=KB
TOP_K=3
ANSWER_CONFIDENCE_THRESHOLD=0.45
```

---

## ✅ Running the Bot

### Start FastAPI App
```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

### Expose via Ngrok
```bash
ngrok http 8000
```
Copy the **Forwarding URL** and set it as your webhook in **Meta Developer Dashboard**.

---

## ✅ WhatsApp Cloud API Integration

### Sending Text Messages
Example in `apps/whatsapp_gateway/send.py`:
```python
def send_whatsapp_text(to: str, text: str):
    url = f"https://graph.facebook.com/v20.0/{PHONE_ID}/messages"
    headers = {"Authorization": f"Bearer {META_TOKEN}", "Content-Type": "application/json"}
    payload = {
        "messaging_product": "whatsapp",
        "to": to,
        "type": "text",
        "text": {"body": text}
    }
    requests.post(url, headers=headers, json=payload)
```

### Sending Media (Images, PDFs)
```python
def send_whatsapp_media(to: str, media_url: str, media_type="image"):
    url = f"https://graph.facebook.com/v20.0/{PHONE_ID}/messages"
    headers = {"Authorization": f"Bearer {META_TOKEN}", "Content-Type": "application/json"}
    payload = {
        "messaging_product": "whatsapp",
        "to": to,
        "type": media_type,
        media_type: {"link": media_url}
    }
    requests.post(url, headers=headers, json=payload)
```

---

## ✅ Adding a New WhatsApp Number
1. Go to **Meta Business Manager → WhatsApp Accounts → Add Phone Number**.
2. Verify via SMS or voice call.
3. Update `.env` with the new `WHATSAPP_PHONE_ID`.

---

## ✅ Troubleshooting
- **Ngrok not recognized**: Ensure ngrok is installed and added to PATH.
- **Webhook not validating**: Check `META_VERIFY_TOKEN` matches your Meta settings.
- **401 Unauthorized**: Verify `META_WHATSAPP_TOKEN` is correct and not expired.
- **No response on WhatsApp**: Confirm webhook URL is active and reachable.

---

## ✅ Deployment Tips
- Use **Docker** for containerized deployment:
  ```bash
  docker-compose up --build
  ```
- Use **systemd** or **PM2** for process management in production.
- Rotate tokens regularly for security.

---

## ✅ License
Apache 2.0
