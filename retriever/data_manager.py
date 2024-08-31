from enum import Enum
import requests
import os

### Data types ###

class DataType(Enum):
    TEXT = 1
    WEB = 2
    PDF = 3
    CSV = 4

### Data class ###

class Data():
    def __init__(self, path, data_type: DataType, chunk_size:int, chunk_overlap:int, extra="None"):
        self.path = path
        self.data_type = data_type
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.extra = extra
    
    def __eq__(self, value: object) -> bool:
        if not isinstance(value, Data):
            return False
        if self.path != value.path:
            return False
        if self.data_type != value.data_type:
            return False
        if self.chunk_size != value.chunk_size:
            return False
        if self.chunk_overlap != value.chunk_overlap:
            return False
        if self.extra != value.extra:
            return False
        return True

    def __ne__(self, value: object) -> bool:
        return not self.__eq__(value)

class DataList():
    def __init__(self, config: dict):
        self.data = []
        self.config = config
        self.main_dir = config['paths']['data']
    
    def get_data_type(self, path) -> DataType:
        """
        Get the type of the data
        
        Args:
            path (str): Path of the data
        
        Returns:
            DataType: Type of the data
        """
        if path.endswith(".txt"):
            return DataType.TEXT
        if path.endswith(".pdf"):
            return DataType.PDF
        if path.endswith(".csv"):
            return DataType.CSV
        if path.startswith("http"):
            return DataType.WEB
        return None
        
    def add(self, path, chunk_size=0, chunk_overlap=0, extra="None") -> None:
        """
        Add a single data file
        
        Args:
            path (str): Path of the file
        """
        data_type = self.get_data_type(path)
        d = Data(path, data_type, chunk_size, chunk_overlap, extra)
        self.data.append(d)
    
    def add_dir(self, path="", chunk_size=0, chunk_overlap=0, extra="None") -> None:
        """
        Add all the files in a directory
        
        Args:
            path (str): Path of the directory
        """
        for file in os.listdir(self.main_dir + path):
            self.add(path + file, chunk_size, chunk_overlap, extra)
    
    def test(self) -> bool:
        """
        Check if data is valid

        Returns:
            bool: True if data is valid, False otherwise
        """
        if not self.data:
            print("\33[1;31m[DataTester]\33[0m: Nessun dato presente")
            return False
        for d in self.data:
            if d.data_type == DataType.TEXT or d.data_type == DataType.PDF or d.data_type == DataType.CSV:
                path = self.main_dir + d.path
                if not os.path.exists(path):
                    print(f"\33[1;31m[DataTester]\33[0m: Il file {d.path} non esiste")
                    return False
                
            if d.data_type == DataType.WEB:
                if requests.head(d.path).status_code != 200:
                    print(f"\33[1;31m[DataTester]\33[0m: Il sito {d.path} non è raggiungibile")
                    return False
                if not d.path.startswith("http"):
                    print(f"\33[1;31m[DataTester]\33[0m: Il path {d.path} non è un URL valido")
                    return False
                if d.extra == "None":
                    print(f"\33[1;31m[DataTester]\33[0m: Il path {d.path} non ha fornito una classe")
                    return False
                
            if d.data_type == DataType.CSV and d.extra == "None":
                print(f"\33[1;31m[DataTester]\33[0m: Il campo extra per il file {d.path} è vuoto")
                return False
            
        print("\33[1;32m[DataTester]\33[0m: Dati validati con successo")
        return True
    
    def print_data(self) -> None:
        """
        Print the data
        """
        for d in self.data:
            print(f"Path: {d.path}, Type: {d.data_type}, Chunk Size: {d.chunk_size}, Chunk Overlap: {d.chunk_overlap}, Extra: {d.extra}")
    
    def get_data(self) -> list:
        """
        Get the data
        
        Returns:
            list: List of data
        """
        return self.data