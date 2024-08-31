from langchain_core.documents import Document
from langchain.retrievers import ContextualCompressionRetriever
from vectorstore import CustomStore
from langchain_cohere import CohereRerank, CohereEmbeddings

class Retriever():
    def __init__(self, config):
        self.threshold = config['threshold']

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
        print("\33[1;36m[Retriever]\33[0m: Risultati recuperati")
        for num, d in enumerate(docs):
            break
            print(f"{num+1}.")
            print(d)
        if threshold != 0:
            return self.filter(docs, threshold)
        else:
            return docs
    
    def filter(self, docs: list[Document], threshold: float) -> list[Document]:
        for d in docs:
            print(d.metadata.get('relevance_score'))
        return [d for d in docs if d.metadata.get('relevance_score') > threshold]
    
    def find_similar(self, doc: Document, threshold=0) -> list[Document]:
        embedded_doc = self.embedder.embed_documents([doc.page_content])[0]
        similar = self.vectorstore.similarity_search_with_score_by_vector(embedded_doc)
        print("\33[1;36m[Retriever]\33[0m: Documento simile recuperato")
        for num, d in enumerate(similar):
            print(f"{num+1}.")
            print(d)
        return similar