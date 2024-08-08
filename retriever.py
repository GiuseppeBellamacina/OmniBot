from langchain_milvus import Milvus
from initializer import Initializer
from langchain_core.documents import Document

class Retriever():
    def __init__(self, config, data):
        self.config = config
        self.threshold = self.config['retriever']['threshold']
        self.simplifier = self.config['retriever']['simplifier']
        self.content_store, title_data = Initializer(self.config, self.config['type_of_data']['content']).initialize(data)
        self.title_store, _ = Initializer(self.config, self.config['type_of_data']['titles']).initialize(title_data)
        print("\33[1;36m[Retriever]\33[0m: Retriever inizializzato")
    
    def retrieve(self, query: str, is_semplified=False) -> list[Document]:
        threshold = self.threshold if not is_semplified else self.threshold * self.simplifier
        ids = [doc.metadata['id'] for doc in self.title_store.similarity_search(query)]
        title_docs = self.content_store.get_by_ids(ids)
        content_docs = self.retrieve_relevant_docs(self.content_store, query, threshold)

        relevant_docs = title_docs + content_docs
        unique_docs = {}
        for doc in relevant_docs:
            unique_docs[doc.metadata['id']] = doc
        print("\33[1;32m[Retriever]\33[0m: Trovati", len(unique_docs), "documenti rilevanti")
        for doc in list(unique_docs.values()):
            print(doc, '\n')
        return list(unique_docs.values())
    
    def retrieve_relevant_docs(vectorstore: Milvus, user_input, threshold) -> list[Document]:
        docs = vectorstore.similarity_search_with_relevance_scores(user_input)
        relevant_docs = [doc[0] for doc in docs if doc[1] > threshold]
        unique_docs = {}
        for doc in relevant_docs:
            unique_docs[doc.metadata['id']] = doc
        return list(unique_docs.values())