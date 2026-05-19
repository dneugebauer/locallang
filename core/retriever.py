from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda

import config


def build_retriever(vectorstore: Chroma, all_docs: list[Document], filter: dict | None = None):
    chroma_kwargs: dict = {"k": config.NUM_RETRIEVED_CHUNKS}
    if filter:
        chroma_kwargs["filter"] = filter

    chroma_retriever = vectorstore.as_retriever(search_kwargs=chroma_kwargs)

    # Mirror the filter on BM25 so both retrievers search the same document scope.
    bm25_docs = _filter_docs(all_docs, filter) if filter else all_docs
    bm25_retriever = BM25Retriever.from_documents(bm25_docs)
    bm25_retriever.k = config.NUM_RETRIEVED_CHUNKS

    ensemble = EnsembleRetriever(
        retrievers=[chroma_retriever, bm25_retriever],
        weights=[config.SEMANTIC_WEIGHT, config.KEYWORD_WEIGHT],
    )

    return ensemble | RunnableLambda(lambda docs: docs[:config.NUM_RETRIEVED_CHUNKS])


def _filter_docs(docs: list[Document], filter: dict) -> list[Document]:
    """Apply simple equality filters (flat key=value) to a document list."""
    return [d for d in docs if all(d.metadata.get(k) == v for k, v in filter.items())]
