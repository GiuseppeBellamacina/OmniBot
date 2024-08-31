from typing import List, Sequence
from langchain_community.vectorstores import FAISS
from langchain_community.docstore import InMemoryDocstore
from langchain_core.documents import Document

class CustomStore(FAISS):
    def __init__(self, embedding_function, index, docstore: InMemoryDocstore, index_to_docstore_id):
        super().__init__(
            embedding_function,
            index,
            docstore,
            index_to_docstore_id
        )
    
    def get_by_ids(self, ids: Sequence[str]) -> List[Document]:
        return [self.docstore.search(doc_id) for doc_id in ids]

    def save(self, path: str):
        self.save_local(path)

    @classmethod
    def get_local(self, path: str, embeddings):
        return self.load_local(path, embeddings=embeddings, allow_dangerous_deserialization=True)