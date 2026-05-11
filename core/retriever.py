from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda

import config


def build_retriever(vectorstore: Chroma, all_docs: list[Document]):
    chroma_retriever = vectorstore.as_retriever(
        search_kwargs={"k": config.NUM_RETRIEVED_CHUNKS}
    )

    bm25_retriever = BM25Retriever.from_documents(all_docs)
    bm25_retriever.k = config.NUM_RETRIEVED_CHUNKS

    ensemble = EnsembleRetriever(
        retrievers=[chroma_retriever, bm25_retriever],
        weights=[config.SEMANTIC_WEIGHT, config.KEYWORD_WEIGHT],
    )

    return ensemble | RunnableLambda(lambda docs: docs[:config.NUM_RETRIEVED_CHUNKS])
