from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_cohere import CohereRerank, CohereEmbeddings
from typing import Any, List
from langchain_core.retrievers import BaseRetriever, RetrieverLike
from langchain_core.callbacks import (
    AsyncCallbackManagerForRetrieverRun,
    CallbackManagerForRetrieverRun,
)
from langchain.retrievers.document_compressors.base import (
    BaseDocumentCompressor,
)

from debugger import debug

class Retriever(BaseRetriever):
    compressor: BaseDocumentCompressor
    retriever: RetrieverLike
    embedder: CohereEmbeddings
    vectorstore: FAISS
    retrieval_threshold: float
    distance_threshold: float
    config: dict

    class Config: arbitrary_types_allowed = True
    
    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
        **kwargs: Any,
    ) -> List[Document]:
        """Get documents relevant for a query.

        Args:
            query: string to find relevant documents for

        Returns:
            Sequence of relevant documents
        """
        docs = self.retriever.invoke(
            query, config={"callbacks": run_manager.get_child()}, **kwargs
        )
        if docs:
            compressed_docs = self.compressor.compress_documents(
                docs, query, callbacks=run_manager.get_child()
            )
            filtered_docs = self.filter_by_similarity(compressed_docs, self.retrieval_threshold)
            if filtered_docs:
                similar_docs = self.search_by_vector(filtered_docs)
                reranked_docs = self.compressor.compress_documents(
                    similar_docs, query, callbacks=run_manager.get_child()
                )
                filtered_docs = self.filter_by_similarity(reranked_docs, self.retrieval_threshold)
                return sorted(filtered_docs, key=lambda x: x.metadata.get('id'))
        else:
            return []
        
    async def _aget_relevant_documents(
        self,
        query: str,
        *,
        run_manager: AsyncCallbackManagerForRetrieverRun,
        **kwargs: Any,
    ) -> List[Document]:
        """Get documents relevant for a query.

        Args:
            query: string to find relevant documents for

        Returns:
            List of relevant documents
        """
        docs = await self.retriever.ainvoke(
            query, config={"callbacks": run_manager.get_child()}, **kwargs
        )
        if docs:
            compressed_docs = await self.compressor.acompress_documents(
                docs, query, callbacks=run_manager.get_child()
            )
            return list(compressed_docs)
        else:
            return []
    
    @debug()
    def filter_by_similarity(self, docs: list[Document], threshold=0) -> list[Document]:
        if threshold == 0:
            return docs
        return [d for d in docs if d.metadata.get('relevance_score') > threshold]
    
    @debug()
    def filter_by_distance(self, docs: list[Document], threshold=0) -> list[Document]:
        if threshold == 0:
            return [d for (d, _) in docs]
        return [d for (d, score) in docs if score < threshold]
    
    @debug()
    def search_by_vector(self, docs: List[Document]) -> list[Document]:
        embedded_docs = self.embedder.embed_documents([d.page_content for d in docs])
        similar_docs = []
        for doc in embedded_docs:
            sim = self.vectorstore.similarity_search_with_score_by_vector(doc)
            if sim:
                similar_docs.extend(sim)
        if similar_docs:
            return self.filter_by_distance(similar_docs, self.distance_threshold)
        return []
    
class RetrieverBuilder():
    def __init__(self, config: dict):
        self.config = config

    def build(self) -> Retriever:
        retrieval_threshold = self.config['retrieval_threshold']
        distance_threshold = self.config['distance_threshold']
        embedder = CohereEmbeddings(model=self.config['embedder'])
        vectorstore = FAISS.load_local(self.config['db'], embeddings=embedder, allow_dangerous_deserialization=True)
        retriever = vectorstore.as_retriever(search_type='similarity', search_kwargs={'k': self.config['k']})
        compressor = CohereRerank(model=self.config['reranker'], top_n=self.config['top_n'])
        print("\33[1;34m[RetrieverBuilder]\33[0m: Retriever inizializzato")
        
        return Retriever(
            compressor=compressor,
            retriever=retriever,
            embedder=embedder,
            vectorstore=vectorstore,
            retrieval_threshold=retrieval_threshold,
            distance_threshold=distance_threshold,
            config=self.config
        )