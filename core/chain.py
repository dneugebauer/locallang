from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.documents import Document

import config

SYSTEM_PROMPT = """You are a helpful assistant that answers questions based strictly on the provided context.

Rules:
- Base your answer on the context below. Do not introduce facts not present in the context.
- When performing revenue or financial calculations, only use figures explicitly labeled as pricing, fees, or retainer amounts. Do not use descriptive figures (e.g. a client's own business size or revenue) as inputs to unrelated calculations.
- When numbers, prices, or figures are present in the context, you must reason over them — perform calculations, derive totals, and show your work step by step when asked for breakdowns.
- Only say you don't have enough information if the relevant facts are genuinely absent from the context. Never refuse to calculate when the numbers are available.
- Be concise and precise.
- Do not list sources in your response.

Context:
{context}"""

_prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "{question}"),
])


def format_docs(docs: list[Document]) -> str:
    parts = []
    for doc in docs:
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page")
        label = f"{source}, page {page}" if page else source
        parts.append(f"[Source: {label}]\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


def build_llm(model: str | None = None):
    return ChatOllama(
        model=model or config.LLM_MODEL,
        base_url=config.OLLAMA_BASE_URL,
    )


def make_messages(context: str, question: str, history: list = []):
    messages = _prompt.format_messages(context=context, question="[see history]")
    # Replace the placeholder human message with full history + current question
    messages = messages[:-1]  # drop the placeholder
    for human, assistant in history:
        messages.append(HumanMessage(content=human))
        messages.append(AIMessage(content=assistant))
    messages.append(HumanMessage(content=question))
    return messages
