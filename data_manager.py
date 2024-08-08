from enum import Enum
import requests
import csv
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


### Data Tester ###

class TestResult(Enum):
    CREATE = 1
    GET = 2
    ERROR = 3


### Data Tester ###

class DataTester():
    def __init__(self, config: dict, type_of_data: str):
        self.config = config
        self.type = type_of_data
        print("\33[1;36m[DataTester]\33[0m: DataTester inizializzato per", self.type)
    
    def test(self, data: list) -> TestResult:
        """
        Check if data is valid and says if the data file and the db file are updated and valid
        
        Returns:
            TestResult: Test result
        """
        print("\33[1;34m[DataTester]\33[0m: Inizio test dei dati")
        valid_data = self.check_data(data)
        if valid_data:
            # check if exists a db
            if os.path.exists(self.config['paths'][self.type]['db']):
                old_data = self.load_data_file()
                # check if data file contains the same data
                if data == old_data:
                    print("\33[1;32m[DataTester]\33[0m: I dati del DB corrispondono")
                    return TestResult.GET
                else:
                    print("\33[1;33m[DataTester]\33[0m: I dati non corrispondono, ricreazione del DB")
                    self.create_data_file(data)
                    return TestResult.CREATE
            else:
                self.create_data_file(data)
                return TestResult.CREATE
        else:
            print(f"\33[1;31m[DataTester]\33[0m: I dati forniti non sono validi")
            return TestResult.ERROR


    def check_data(self, data) -> bool:
        """
        Check if data is valid

        Returns:
            bool: True if data is valid, False otherwise
        """
        data_dir = self.config['paths'][self.type]['data']
        for d in data:
            if d.data_type == DataType.TEXT or d.data_type == DataType.PDF or d.data_type == DataType.CSV:
                path = data_dir + d.path
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
    
    def create_data_file(self, data) -> None:
        """
        Create a file used to store the information of the list of data
        """
        try:
            data_tester = self.config['paths'][self.type]['data_tester']
            # create directory if not exists
            if not os.path.exists(os.path.dirname(data_tester)):
                os.makedirs(os.path.dirname(data_tester))
            with open(data_tester, "w", newline="") as file:
                writer = csv.writer(file)
                for d in data:
                    writer.writerow([d.path, d.data_type.value, d.chunk_size, d.chunk_overlap, d.extra])
                    
            print(f"\33[1;32m[DataTester]\33[0m: Data File creato e salvato in {data_tester}")
        
        except Exception as e:
            print(f"\33[1;31m[DataTester]\33[0m: Errore durante la creazione del Data File: {e}")
            raise e
    
    def load_data_file(self) -> list:
        """
        Verify the data files
        
        Returns:
            list: List of data
        """
        data = []
        data_tester = self.config['paths'][self.type]['data_tester']
        if not os.path.exists(data_tester):
            print(f"\33[1;31m[DataTester]\33[0m: File {data_tester} non trovato")
            return data
        
        try:
            with open(data_tester, "r") as file:
                reader = csv.reader(file)
                for row in reader:
                    path, data_type, chunk_size, chunk_overlap, extra = row
                    data.append(Data(path, DataType(int(data_type)), int(chunk_size), int(chunk_overlap), extra))

            print(f"\33[1;32m[DataTester]\33[0m: Info sui dati caricati dal file {data_tester}")
            return data

        except Exception as e:
            print(f"\33[1;31m[DataTester]\33[0m: Errore durante il caricamento del Data File: {e}")
            raise e

class DataList():
    def __init__(self):
        self.data = []
        print("\33[1;36m[DataList]\33[0m: DataList creata")
    
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
        
    def add(self, path: str, chunk_size=0, chunk_overlap=0, extra="None") -> None:
        """
        Add a single data file
        
        Args:
            path (str): Path of the file
        """
        data_type = self.get_data_type(path)
        d = Data(path, data_type, chunk_size, chunk_overlap, extra)
        self.data.append(d)
    
    def add_dir(self, main_dir: str, path: str, chunk_size=0, chunk_overlap=0, extra="None") -> None:
        """
        Add all the files in a directory
        
        Args:
            path (str): Path of the directory
        """
        for file in os.listdir(main_dir + path):
            self.add(path + file, chunk_size, chunk_overlap, extra)
    
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