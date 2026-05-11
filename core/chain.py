from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document

import config

SYSTEM_PROMPT = """You are a helpful assistant that answers questions based strictly on the provided context.

Rules:
- Answer only from the context below. Do not use prior knowledge.
- If the context does not contain enough information to answer, say "I don't have enough information in the indexed documents to answer that."
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


def build_llm():
    return ChatOllama(
        model=config.LLM_MODEL,
        base_url=config.OLLAMA_BASE_URL,
    )


def make_messages(context: str, question: str):
    return _prompt.format_messages(context=context, question=question)
