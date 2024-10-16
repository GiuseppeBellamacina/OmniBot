from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from langchain_core.messages import HumanMessage, AIMessage

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
            full_text = self.message.content + " " + " ".join(doc.page_content for doc in self.documents)
            self.vector = vectorizer.transform([full_text]).toarray()[0]
        return self.vector

class ChatHistory():
    def __init__(self):
        self.messages = []
        self.vectorizer = None
    
    def add_message_from_user(self, user_input: str): # * ste funzioni vanno chiamate dopo che il modello ha finito di rispondere
        message = MessageWithDocs(
            message = HumanMessage(content = user_input),
            documents = []
        )
        self.messages.append(message)
    
    def add_message_from_response(self, response: dict):
        message = MessageWithDocs(
            message = AIMessage(content = response.get('answer', '')),
            documents = response.get('context', '')
        )
        self.messages.append(message)
    
    def train_vectorizer(self):
        all_texts = []
        for msg in self.messages:
            if msg.documents != []:
                all_texts.append(msg.message.content + " " + " ".join(doc.page_content for doc in msg.documents))
        self.vectorizer = TfidfVectorizer(encoding='utf-8').fit(all_texts)
    
    def get_message_ctx(self, message: str, threshold: float):
        message = MessageWithDocs(
            message = HumanMessage(content = message),
            documents = []
        )
        selected_vector = message.embed_self(self.vectorizer)
        ctx = []
        for i in range(len(self.messages)):
            vector = self.messages[i].embed_self(self.vectorizer)
            similarity = cosine_similarity([selected_vector], [vector])[0][0]
            if similarity > threshold:
                ctx.extend(self.messages[i].documents)
        return ctx
    
    def get_last_message_ctx(self, threshold: float):
        selected_vector = self.messages[-1].embed_self(self.vectorizer)
        ctx = []
        for i in range(len(self.messages) - 1):
            vector = self.messages[i].embed_self(self.vectorizer)
            similarity = cosine_similarity([selected_vector], [vector])[0][0]
            if similarity > threshold:
                ctx.extend(self.messages[i].documents)
        ctx.extend(self.messages[-1].documents)
        return ctx
    
    def get_followup_ctx(self, message: str, threshold: float):
        self.train_vectorizer()
        message_ctx = self.get_message_ctx(message, threshold)
        if message_ctx:
            return message_ctx
        else:
            return self.get_last_message_ctx(threshold)
    
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
    def __init__(self):
        self.containers = None
        self.text = ""
        self.time = 0

    def start(self, containers=None):
        self.time = time()
        self.text = ""
        self.containers = containers

    def on_new_token(self, token: dict) -> None:
        token = token.get('answer', None)
        if token:
            self.text += token
            self.containers[0].markdown(self.text)

    def end(self):
        self.text = ""
        self.time = time() - self.time
        text_time = f"⏱ Tempo di risposta: {self.time:.2f} secondi"
        self.containers[1].markdown(text_time)
    
    def error(self, error: Exception):
        self.time = time() - self.time
        self.containers[0].markdown(f"Errore: {error}")
        self.containers[1].markdown(f"⏱ Tempo di risposta: {self.time:.2f} secondi")
        self.text = ""
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

def format_docs(docs):
    return "\n\n".join([d.page_content for d in docs])