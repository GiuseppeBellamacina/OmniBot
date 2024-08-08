from data_manager import Data, DataTester, TestResult
from splitter import Splitter, Titler
from vectorizer import Vectorizer
import os

class Initializer():
    """
    Initializer class: manages the initialization of the data using
    the DataTester, Splitter and Vectorizer classes.
    """
    def __init__(self, config: dict, type_of_data: str):
        self.config = config
        self.type = type_of_data
        print("\33[1;36m[Initializer]\33[0m: Inizializzatore inizializzato per", self.type)
    
    def initialize(self, data: list[Data]):
        """
        Initialize the data using the DataTester, Splitter and Vectorizer classes.
        
        Returns:
            Milvus: Vector store
        """
        print("\33[1;34m[Initializer]\33[0m: Avvio inizializzazione")
        if self.type != "titles":
            data_tester = DataTester(self.config, self.type)
            test_result = data_tester.test(data)
        else:
            if os.path.exists(self.config['paths']['titles']['db']):
                test_result = TestResult.GET
            else:
                test_result = TestResult.CREATE
        
        if test_result == TestResult.CREATE:
            splitter = Splitter(self.config, self.type)
            chunks = splitter.create_chunks(data)
            title_data = None
            if self.type != "titles":
                title_data = Titler(self.config).create_title_file(chunks)
            vectorizer = Vectorizer(self.config, self.type)
            return vectorizer.create_db(chunks), title_data
        
        elif test_result == TestResult.GET:
            vectorizer = Vectorizer(self.config, self.type)
            return vectorizer.get_db()
        
        elif test_result == TestResult.ERROR:
            return None