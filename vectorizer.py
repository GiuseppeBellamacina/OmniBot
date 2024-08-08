from langchain_huggingface import HuggingFaceEmbeddings
from langchain_milvus import Milvus
from langchain_core.documents import Document

from tqdm import tqdm

import torch
import shutil
import os

class Vectorizer():
    def __init__(self, config: dict, type_of_data: str):
        self.config = config
        self.type = type_of_data
        print("\33[1;36m[Vectorizer]\33[0m: Vectorizer inizializzato per", self.type)
    
    def create_db(self, chunks: list[Document]) -> Milvus:
        """
        Create vector store

        Args:
            chunks (list): List of Documents chunks

        Returns:
            Milvus: Vector store
        """
        try:
            embedder_name = self.config['embedder']['name']
            if torch.cuda.is_available():
                embedder_kwargs = self.config['embedder']['kwargs_cuda']
            else:
                embedder_kwargs = self.config['embedder']['kwargs_cpu']
            embedder = HuggingFaceEmbeddings(model_name=embedder_name, model_kwargs=embedder_kwargs)
            print(f"\33[1;32m[Vectorizer]\33[0m: Embedder {embedder_name} caricato con i seguenti parametri: {embedder_kwargs}")
            
            print("\33[1;34m[Vectorizer]\33[0m: Avvio creazione DB")
            db = self.config['paths'][self.type]['db']
            if os.path.exists(db):
                shutil.rmtree(db, ignore_errors=True)
                print("\33[1;33m[Vectorizer]\33[0m: DB esistente eliminato")
            vectorstore = Milvus.from_documents(
                documents=chunks,
                embedding=embedder,
                connection_args={"uri": db},
                collection_name=self.config['type_of_data'][self.type]
            )
            print("\33[1;32m[Vectorizer]\33[0m: DB creato e salvato in", db)
            return vectorstore
        
        except Exception as e:
            print(f"\33[1;31m[Vectorizer]\33[0m: Errore durante la creazione del vector store: {e}")
            raise e
        
    def get_db(self) -> Milvus:
        """
        Get vector store

        Returns:
            Milvus: Vector store
        """
        db = self.config['paths'][self.type]['db']
        if not os.path.exists(db):
            print(f"\33[1;31m[Vectorizer]\33[0m: Database non trovato nella posizione {db}")
            return None
        
        try:
            embedder_name = self.config['embedder']['name']
            if torch.cuda.is_available():
                embedder_kwargs = self.config['embedder']['kwargs_cuda']
            else:
                embedder_kwargs = self.config['embedder']['kwargs_cpu']
            embedder = HuggingFaceEmbeddings(model_name=embedder_name, model_kwargs=embedder_kwargs)
            print(f"\33[1;32m[Vectorizer]\33[0m: Embedder {embedder_name} caricato con i seguenti parametri: {embedder_kwargs}")
        
            vectorstore = Milvus(
                connection_args={"uri": db},
                collection_name=self.config['type_of_data'][self.type],
                embedding=embedder
            )
            print("\33[1;32m[Vectorizer]\33[0m: DB caricato")
            return vectorstore
        
        except Exception as e:
            print(f"\33[1;31m[Vectorizer]\33[0m: Errore durante il caricamento del DB: {e}")
            raise e

    def batch(self, chunks, n_max=10000):
        batches = []
        current_batch = []
        count = 0

        for c in chunks:
            chunk_length = len(c.page_content)
            
            if count + chunk_length >= n_max:
                batches.append(current_batch)
                current_batch = [c]
                count = chunk_length
            else:
                current_batch.append(c)
                count += chunk_length

        if current_batch:
            batches.append(current_batch)
        
        return batches