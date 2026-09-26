import requests
from fastapi import FastAPI, Request
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# ---------------------------------------------------------
# Fill these in directly for a quick local test.
# For real deployment, switch these to os.environ["..."] instead
# so the keys aren't sitting in plain text in your code.
# ---------------------------------------------------------
import os

BOT_TOKEN = os.environ["BOT_TOKEN"]
GROQ_API_KEY = os.environ["GROQ_API_KEY"]

TELEGRAM_API_URL = f"https://api.telegram.org/bot{BOT_TOKEN}"

app = FastAPI()

# LLM chain — built once, reused for every incoming message
llm = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model="openai/gpt-oss-120b",
    temperature=0,
)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI assistant."),
    ("human", "{input}"),
])

parser = StrOutputParser()
chain = prompt | llm | parser


def send_message(chat_id: int, text: str):
    url = f"{TELEGRAM_API_URL}/sendMessage"
    response = requests.post(url, json={"chat_id": chat_id, "text": text})
    return response.json()


@app.post("/telegram-webhook")
async def telegram_webhook(request: Request):
    data = await request.json()

    message = data.get("message")
    if not message or "text" not in message:
        return {"ok": True}

    chat_id = message["chat"]["id"]
    user_text = message["text"]

    try:
        reply_text = chain.invoke({"input": user_text})
    except Exception as e:
        reply_text = f"Sorry, something went wrong: {e}"

    send_message(chat_id, reply_text)
    return {"ok": True}


@app.get("/")
async def health_check():
    return {"status": "running"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)