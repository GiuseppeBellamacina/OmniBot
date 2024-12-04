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
        callbacks = run_manager.get_child()
        docs = self.retriever.invoke(query, config={"callbacks": callbacks}, **kwargs)
        if not docs:
            return []

        compressed_docs = self.compressor.compress_documents(docs, query, callbacks=callbacks)
        if not compressed_docs:
            return []

        filtered_docs = self.filter_by_similarity(compressed_docs, self.retrieval_threshold)
        if not filtered_docs:
            return []

        similar_docs = self.search_by_vector(filtered_docs)
        if not similar_docs:
            return []

        reranked_docs = self.compressor.compress_documents(similar_docs, query, callbacks=callbacks)
        if not reranked_docs:
            return []

        refiltered_docs = self.filter_by_similarity(reranked_docs, self.retrieval_threshold)
        if not refiltered_docs:
            return []

        return sorted(refiltered_docs, key=lambda x: x.metadata.get('id'))
        
    async def _aget_relevant_documents(
        self,
        query: str,
        *,
        run_manager: AsyncCallbackManagerForRetrieverRun,
        **kwargs: Any,
    ) -> List[Document]:
        """Get documents relevant for a query asynchronously.

        Args:
            query: string to find relevant documents for

        Returns:
            List of relevant documents
        """
        callbacks = run_manager.get_child()

        # Invoca il retriever in modo asincrono
        docs = await self.retriever.ainvoke(query, config={"callbacks": callbacks}, **kwargs)
        if not docs:
            return []

        # Comprime i documenti in modo asincrono
        compressed_docs = await self.compressor.acompress_documents(docs, query, callbacks=callbacks)
        if not compressed_docs:
            return []

        # Filtra i documenti per similarità in modo asincrono
        filtered_docs = await self.afilter_by_similarity(compressed_docs, self.retrieval_threshold)
        if not filtered_docs:
            return []

        # Cerca i documenti nel vettore in modo asincrono
        similar_docs = await self.asearch_by_vector(filtered_docs)
        if not similar_docs:
            return []

        # Rerank dei documenti compressi
        reranked_docs = await self.compressor.acompress_documents(similar_docs, query, callbacks=callbacks)
        if not reranked_docs:
            return []

        # Filtra nuovamente per similarità
        refiltered_docs = await self.afilter_by_similarity(reranked_docs, self.retrieval_threshold)
        if not refiltered_docs:
            return []

        # Ritorna i documenti ordinati per ID
        return sorted(refiltered_docs, key=lambda x: x.metadata.get('id'))

    def filter_by_similarity(self, docs: list[Document], threshold=0) -> list[Document]:
        if threshold == 0:
            return docs
        return [d for d in docs if d.metadata.get('relevance_score') > threshold]

    async def afilter_by_similarity(self, docs: list[Document], threshold=0) -> list[Document]:
        if threshold == 0:
            return docs
        return [d for d in docs if d.metadata.get('relevance_score') > threshold]
    
    def filter_by_distance(self, docs: list[Document], threshold=0) -> list[Document]:
        if threshold == 0:
            return [d for (d, _) in docs]
        return [d for (d, score) in docs if score < threshold]
    
    async def afilter_by_distance(self, docs: list[Document], threshold=0) -> list[Document]:
        if threshold == 0:
            return [d for (d, _) in docs]
        return [d for (d, score) in docs if score < threshold]
    
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

    async def asearch_by_vector(self, docs: List[Document]) -> list[Document]:
        embedded_docs = await self.embedder.aembed_documents([d.page_content for d in docs])
        similar_docs = []
        for doc in embedded_docs:
            sim = await self.vectorstore.asimilarity_search_with_score_by_vector(doc)
            if sim:
                similar_docs.extend(sim)
        if similar_docs:
            return await self.afilter_by_distance(similar_docs, self.distance_threshold)
        return []
    
class RetrieverBuilder():
    @classmethod
    def build(self, config) -> Retriever:
        retrieval_threshold = config['retrieval_threshold']
        distance_threshold = config['distance_threshold']
        embedder = CohereEmbeddings(model=config['embedder'])
        vectorstore = FAISS.load_local(config['db'], embeddings=embedder, allow_dangerous_deserialization=True)
        retriever = vectorstore.as_retriever(search_type='similarity', search_kwargs={'k': config['k']})
        compressor = CohereRerank(model=config['reranker'], top_n=config['top_n'])
        print("\33[1;34m[RetrieverBuilder]\33[0m: Retriever inizializzato")
        
        return Retriever(
            compressor=compressor,
            retriever=retriever,
            embedder=embedder,
            vectorstore=vectorstore,
            retrieval_threshold=retrieval_threshold,
            distance_threshold=distance_threshold,
            config=config
        )