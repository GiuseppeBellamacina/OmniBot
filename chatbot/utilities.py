from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.documents import Document
from debugger import debug

from time import time
import yaml

###  Messages ###

class MessageWithDocs():
    def __init__(self, message, documents):
        self.message = message
        self.documents = documents
        self.vector = None

    def embed_self(self, vectorizer):
        if self.vector is None:  # Embed only if not already embedded
            full_text = self.message.content
            if self.documents:
                full_text += "\n\n" + docs_to_string(self.documents, sep="")
            self.vector = vectorizer.transform([full_text]).toarray()[0]
        return self.vector

class ChatHistory():
    def __init__(self, limit: int = 0):
        self.messages:  list[MessageWithDocs] = []
        self.vectorizer = None
        self.limit = limit
    
    def limit_history(self):
        if self.limit != 0:
            self.messages = self.messages[-self.limit:] # lascio solo gli ultimi messaggi
    
    def add_message_from_user(self, user_input: dict): # * ste funzioni vanno chiamate dopo che il modello ha finito di rispondere
        message = MessageWithDocs(
            message = HumanMessage(content = user_input.get('input', '')),
            documents = []
        )
        self.messages.append(message)
        self.limit_history()
    
    def add_message_from_response(self, response: dict):
        response_context = response.get('context', [])
        if response_context:
            response_context = string_to_docs(response_context)
        message = MessageWithDocs(
            message = AIMessage(content = response.get('answer', '')),
            documents = response_context
        )
        print(f"\33[1;31m[DEBUG]\33[0m: {message.documents}")
        self.messages.append(message)
        self.limit_history()
    
    @debug()
    def train_vectorizer(self):
        all_texts = []
        ai_messagges = [msg for msg in self.messages if isinstance(msg.message, AIMessage)]
        for msg in ai_messagges:
            all_texts.append(msg.message.content + "\n\n" + docs_to_string(msg.documents, sep=""))
        print(f"\33[1;31m[DEBUG]\33[0m: {all_texts}")
        if all_texts:
            self.vectorizer = TfidfVectorizer(encoding='utf-8').fit(all_texts)
    
    @debug()
    def get_old_messages_ctx(self, threshold: float):
        print(f"\33[1;36m[DEBUG]\33[0m: Sto per embeddare il primo messaggio")
        user_message_vector = self.messages[-1].embed_self(self.vectorizer)
        ctx = []
        ai_messagges = [msg for msg in self.messages if isinstance(msg.message, AIMessage)]
        print(f"\33[1;36m[DEBUG]\33[0m: Ci sono {len(ai_messagges)} messaggi da embeddare")
        count = 0
        for msg in ai_messagges:
            print(f"\33[1;36m[DEBUG]\33[0m: Sto per embeddare il messaggio {count}")
            count += 1
            vector = msg.embed_self(self.vectorizer)
            similarity = cosine_similarity([user_message_vector], [vector])[0][0]
            if similarity > threshold:
                ctx.extend(msg.documents)
        if not ctx: # Se non ho trovato nessun contesto, prendo l'ultimo contesto
            print(f"\33[1;36m[DEBUG]\33[0m: Non ho trovato nessun contesto, prendo l'ultimo")
            ctx.extend(ai_messagges[-1].documents)
        return ctx
    
    @debug()
    def get_followup_ctx(self, threshold: float):
        self.train_vectorizer()
        if self.vectorizer is None:
            return []
        return self.get_old_messages_ctx(threshold)
    
    def get_all_messages(self):
        if self.messages == []:
            return []
        return [msg.message for msg in self.messages]

    def get_last_messages(self, n: int):
        if self.messages == []:
            return []
        if n > len(self.messages):
            n = len(self.messages)
        return [msg.message for msg in self.messages[-n:]]

    def clear(self):
        self.messages = []
        self.vectorizer = None

### Handler ###

class StdOutHandler():
    """
    Class to manage token's stream
    """
    def __init__(self, debug=False):
        self.containers = None
        self.text = ""
        self.time = 0
        self.debug = debug

    def start(self, containers=None):
        self.time = time()
        self.text = ""
        self.containers = containers

    def on_new_token(self, token: dict) -> None:
        token = token.get('answer', None)
        if self.debug:
            if token:
                print(token, sep="", end="", flush=True)
        if token:
            self.text += token
            if self.containers:
                self.containers[0].markdown(self.text)

    def end(self):
        self.text = ""
        self.time = time() - self.time
        text_time = f"⏱ Tempo di risposta: {self.time:.2f} secondi"
        if self.debug:
            print(text_time)
        if self.containers:
            self.containers[1].markdown(text_time)
    
    def error(self, error: Exception):
        self.time = time() - self.time
        self.text = ""
        if self.debug:
            print(f"\33[1;31m[STDOUTHANDLER]\33[0m: {error}")
            print(f"⏱ Tempo di risposta: {self.time:.2f} secondi")
        if self.containers:
            self.containers[0].markdown(f"Errore: {error}")
            self.containers[1].markdown(f"⏱ Tempo di risposta: {self.time:.2f} secondi")
        raise error

def load_config(file_path='config.yaml'):
    """
    Load configuration file
    
    Args:
        file_path (str): Configuration file path
    
    Returns:
        dict: Configuration file
    """
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def docs_to_string(docs, sep="§§§§§"):
    return f"\n{sep}\n".join([d.page_content for d in docs])

def string_to_docs(string, sep="§§§§§"):
    return [Document(page_content=page) for page in string.split(sep)]