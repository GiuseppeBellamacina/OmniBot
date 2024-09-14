from langchain_core.documents import Document
from langchain.retrievers import ContextualCompressionRetriever
from vectorstore import CustomStore
from langchain_cohere import CohereRerank, CohereEmbeddings
from debugger import debug

class Retriever():
    def __init__(self, config):
        self.threshold = config['retrieval_threshold']

        self.embedder = CohereEmbeddings(model=config['embedder'])
        self.vectorstore = CustomStore.get_local(
            path=config['db'],
            embeddings=self.embedder
        )
        self.retriever = self.vectorstore.as_retriever(search_type='similarity', search_kwargs={'k': 10})
        
        compressor = CohereRerank(model=config['reranker'], top_n=6)
        self.compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=self.retriever
        )
        print("\33[1;36m[Retriever]\33[0m: Retriever inizializzato")
    
    def retrieve(self, query: str, threshold=0) -> list[Document]:
        docs = self.compression_retriever.invoke(query)
        if threshold != 0:
            return self.filter(docs, threshold)
        else:
            return docs
    
    def filter(self, docs: list[Document], threshold: float) -> list[Document]:
        return [d for d in docs if d.metadata.get('relevance_score') > threshold]
    
    def find_similar(self, doc: Document, threshold=0) -> list[Document]:
        embedded_doc = self.embedder.embed_documents([doc.page_content])[0]
        similar_docs = self.vectorstore.similarity_search_with_score_by_vector(embedded_doc)
        if threshold != 0:
            similar_docs = [(d, score) for d, score in similar_docs if score < threshold]
        for d, score in similar_docs:
            d.metadata['relevance_score'] = score
        return [d for d, _ in similar_docs]