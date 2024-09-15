from data_manager import Data
from splitter import Splitter
from langchain_community.vectorstores import VectorStore
from tqdm import tqdm

class DBMaker():
    """
    Create the database containing the vectors of the data.
    """
    def __init__(self, config: dict, vectorstore: VectorStore):
        self.config = config
        self.vectorstore = vectorstore
        print("\33[1;34m[DBMaker]\33[0m: Maker del database inizializzato")
    
    def make(self, data: list[Data]):
        """
        Create the database.
        """
        splitter = Splitter(self.config['paths']['data'])
        docs = splitter.create_chunks(data)
        batches = self.batch(docs)
        for batch in tqdm(batches, desc="Caricamento documenti..."):
            self.vectorstore.add_documents(batch)
        self.vectorstore.save(self.config['paths']['db'])
    
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