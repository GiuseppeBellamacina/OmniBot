from langchain_core.prompts import (
    PromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder
)
from langchain_core.runnables import (
    RunnableBranch,
    RunnableLambda,
    RunnableSequence,
    RunnablePassthrough,
    Runnable
)
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.documents import Document

from retriever import Retriever
from utilities import ChatHistory, StdOutHandler, docs_to_string
from abc import ABC, abstractmethod

class ChainInterface(ABC):
    @abstractmethod
    def run(self):
        pass
    
    @abstractmethod
    def invoke(self, input, containers=None):
        pass
    
    @abstractmethod
    def stream(self, input, containers=None):
        pass
    
    @abstractmethod
    async def ainvoke(self, input, containers=None):
        pass
    
    @abstractmethod
    async def astream(self, input, containers=None):
        pass

class Chain(ChainInterface):
    def __init__(self, llm: Runnable, handler: StdOutHandler | None, name: str = "Chain"):
        self.llm = llm
        self.handler = handler
        self.name = name
    
    def run(self) -> Runnable:
        return self.llm
    
    def invoke(self, input, containers=None):
        try:
            if self.handler:
                self.handler.start(containers)
            response = self.run().invoke(input)
            if self.handler:
                self.handler.on_new_token(response)
                self.handler.end()
            return response
        except Exception as e:
            if self.handler:
                self.handler.error(e)
            else:
                raise e
            return {}
    
    async def ainvoke(self, input, containers=None):
        try:
            if self.handler:
                self.handler.start(containers)
            response = await self.run().ainvoke(input)
            if self.handler:
                self.handler.on_new_token(response)
                self.handler.end()
            return response
        except Exception as e:
            if self.handler:
                self.handler.error(e)
            else:
                raise e
            return {}

    def stream(self, input, containers=None):
        try:
            if self.handler:
                self.handler.start(containers)
            response = {}
            out = self.run().stream(input)
            for token in out:
                if self.handler:
                    self.handler.on_new_token(token)
                response += token
            if self.handler:
                self.handler.end()
            return response
        except Exception as e:
            if self.handler:
                self.handler.error(e)
            else:
                raise e
            return {}
    
    async def astream(self, input, containers=None):
        try:
            if self.handler:
                self.handler.start(containers)
            response = {}
            out = self.run().astream(input)
            async for token in out:
                if self.handler:
                    self.handler.on_new_token(token)
                response += token
            if self.handler:
                self.handler.end()
            return response
        except Exception as e:
            if self.handler:
                self.handler.error(e)
            else:
                raise e
            return {}
    
    def fill_prompt(self, system_template: str):
        return PromptTemplate.from_template(system_template).with_config(run_name="PromptTemplate")

class HistoryAwareChain(Chain):
    def __init__(self, llm: Runnable, handler: StdOutHandler | None, name: str, history: ChatHistory):
        super().__init__(llm, handler, name)
        self.history = history
        
    def get_history_ctx(self):
        return RunnablePassthrough.assign(
            history_ctx = RunnableLambda(lambda x: self.history.get_all_messages())
        ).with_config(run_name="HistoryCTX")
    
    def fill_prompt(self, system_template: str):
        return ChatPromptTemplate.from_messages(
            [
                ("system", system_template),
                MessagesPlaceholder("history_ctx", optional=True)
            ]
        ).with_config(run_name="PromptTemplate")
    
    def invoke(self, input, containers=None):
        try:
            if self.handler:
                self.handler.start(containers)
            self.history.add_message_from_user(input)
            response = self.run().invoke(input)
            if self.handler:
                self.handler.on_new_token(response)
                self.handler.end()
            self.history.add_message_from_response(response)
            return response
        except Exception as e:
            if self.handler:
                self.handler.error(e)
            else:
                raise e
            return {}
    
    async def ainvoke(self, input, containers=None):
        try:
            if self.handler:
                self.handler.start(containers)
            self.history.add_message_from_user(input)
            response = await self.run().ainvoke(input)
            if self.handler:
                self.handler.on_new_token(response)
                self.handler.end()
            self.history.add_message_from_response(response)
            return response
        except Exception as e:
            if self.handler:
                self.handler.error(e)
            else:
                raise e
            return {}

    def stream(self, input, containers=None):
        try:
            if self.handler:
                self.handler.start(containers)
            response = {}
            self.history.add_message_from_user(input)
            out = self.run().stream(input)
            for token in out:
                if self.handler:
                    self.handler.on_new_token(token)
                response += token
            if self.handler:
                self.handler.end()
            self.history.add_message_from_response(response)
            return response
        except Exception as e:
            if self.handler:
                self.handler.error(e)
            else:
                raise e
            return {}
    
    async def astream(self, input, containers=None):
        try:
            if self.handler:
                self.handler.start(containers)
            response = {}
            self.history.add_message_from_user(input)
            out = self.run().astream(input)
            async for token in out:
                if self.handler:
                    self.handler.on_new_token(token)
                response += token
            if self.handler:
                self.handler.end()
            self.history.add_message_from_response(response)
            return response
        except Exception as e:
            if self.handler:
                self.handler.error(e)
            else:
                raise e
            return {}

CONVERSATION_TEMPLATE = """ \
Tu sei un assistente che risponde alle domande relative all'Aeronautica Militare Italiana. \
Rispondi sempre in ITALIANO, e solo alle domande che possono avere a che fare con l'aeronautica:
ESEMPI: piloti, aerei, carriere, accademia, corsi, concorsi, bandi... \
NON rispondere ad una domanda con un'altra domanda. \
Se la domanda NON è inerente al contesto, rispondi con "Non so rispondere a questa domanda". \
Dialoga con l'utente e rispondi alle sue domande. \
Se ti dovessero chiedere il tuo nome, tu ti chiami Azzurra.
Se l'utente ti ringrazia, rispondi con "Prego" o "Non c'è di che" e renditi sempre disponibile. \
Cerca di rispondere in modo adeguato alla conversazione.
"""

class ConversationalChain(HistoryAwareChain):
    def __init__(self, llm: Runnable, handler: StdOutHandler | None, name: str, history: ChatHistory):
        super().__init__(llm, handler, name, history)
        print("\33[1;34m[ConversationalChain]\33[0m: Chain inizializzata")
    
    def sequence(self):
        return RunnableSequence(
            self.fill_prompt(CONVERSATION_TEMPLATE),
            self.llm
        ).with_config(run_name="ConversationSequence")
    
    def answer(self):
        return RunnablePassthrough.assign(
            answer = RunnableLambda(lambda x: self.sequence())
        ).with_config(run_name="ConversationAnswer")
    
    def run(self):
        return ((
            self.get_history_ctx() | self.answer()
        ).assign(signature=lambda x: self.name)
        ).with_config(run_name=self.name)
  
CLASSIFICATION_TEMPLATE = """
Stai parlando con un utente e devi classificare le sue domande in 3 categorie: "summary", "document" e "conversational".

- Le domande "summary" richiedono un riassunto delle informazioni precedenti. Esempi: "Puoi fare un riassunto?", "Riassumi ciò di cui abbiamo parlato.", "Riassumi la conversazione".
- Le domande "document" sono domande che chiedono di argomenti specifici basati su documenti forniti. Esempi: "Qual è la capitale della Francia?", "Come si usa un saldatore?, "Dimmi di più", "Approfondisci questo punto, "Fammi capire meglio", "Perché è così?", "Continua".
- Le domande "conversational" sono domande che NON sono basate su documenti forniti. Esempi: "Ciao", "Che cosa sai fare?", "Come ti chiami?", "Chi ti ha creato?".

DOMANDA:
{input}

Rispondi con un JSON che indica il tipo di domanda.
Esempio: {{"type": "summary"}} o {{"type": "document"}} o {{"type": "conversational"}}
"""

class ClassificationChain(Chain):
    """
    It's the chain which classifies the user's questions in three categories: "summary", "document", and "conversational".
    To use it it is necessary to specify:
    - input
    """
    def __init__(self, llm: Runnable, handler: StdOutHandler | None, name: str):
        super().__init__(llm, handler, name)
        print("\33[1;34m[ClassificationChain]\33[0m: Chain inizializzata")
    
    def sequence(self):
        return RunnableSequence(
            self.fill_prompt(CLASSIFICATION_TEMPLATE),
            self.llm,
            JsonOutputParser()
        ).with_config(run_name="ClassificationSequence")
        
    def run(self):
        return (
            self.sequence()
        ).with_config(run_name=self.name)

SUMMARIZATION_TEMPLATE = """
Stai parlando con un utente e devi fare un riassunto delle informazioni di cui avete discusso. \
NON ripetere l'ultima domanda dell'utente. \
Se possibile rendi la tua risposta strutturata, utilizzando elenco puntato o numerato. \
Rispondi in ITALIANO (o nella lingua della domanda) rispettando la richiesta dell'utente e utilizzando le informazioni seguenti. \
"""

class SummarizationChain(HistoryAwareChain):
    """
    It's the chain which summarizes the information discussed with the user.
    To use it it is necessary to specify:
    - input
    """
    def __init__(self, llm: Runnable, handler: StdOutHandler | None, name: str, history: ChatHistory):
        super().__init__(llm, handler, name, history)
        print("\33[1;34m[SummarizationChain]\33[0m: Chain inizializzata")
        
    def sequence(self):
        return RunnableSequence(
            self.fill_prompt(SUMMARIZATION_TEMPLATE),
            self.llm
        ).with_config(run_name="SummarizationSequence")
        
    def answer(self):
        return RunnablePassthrough.assign(
            answer = RunnableLambda(lambda x: self.sequence())
        ).with_config(run_name="SummarizationAnswer")
    
    def run(self):
        return ((
            self.get_history_ctx() | self.answer()
        ).assign(signature=lambda x: self.name)
        ).with_config(run_name=self.name)
    
RAG_TEMPLATE = """ \
Tu sei un assistente che risponde alle domande relative all'Aeronautica Militare Italiana. \
Rispondi sempre in ITALIANO, e solo alle domande che possono avere a che fare con l'aeronautica:
ESEMPI: piloti, aerei, carriere, accademia, corsi, concorsi, bandi... \
NON rispondere ad una domanda con un'altra domanda. \
Se la domanda NON è inerente al contesto, rispondi con "Non so rispondere a questa domanda". \
L'utente NON deve sapere che stai rispondendo grazie ai seguenti documenti. \
Se possibile rendi la tua risposta strutturata, utilizzando elenco puntato o numerato. \

CONTESTO:
{context}
"""

class RAGChain(HistoryAwareChain):
    """
    It's the chain which manages the RAG questions.
    To use it it is necessary to specify:
    - input
    """
    def __init__(self, llm: Runnable, handler: StdOutHandler | None, name: str, history: ChatHistory,
                 retriever: Retriever, retrieval_threshold: float, followup_threshold: float, distance_threshold: float):
        super().__init__(llm, handler, name, history)
        self.retriever = retriever
        
        self.retrieval_threshold = retrieval_threshold
        self.followup_threshold = followup_threshold
        self.distance_threshold = distance_threshold
        print("\33[1;34m[RAGChain]\33[0m: Chain inizializzata")
    
    def context(self):
        return RunnablePassthrough.assign(
            context = RunnableLambda(lambda x: self.get_ctx(x.get('input')))
        ).with_config(run_name="RAGContext")
    
    def sequence(self):
        return RunnableSequence(
            self.fill_prompt(RAG_TEMPLATE),
            self.llm
        ).with_config(run_name="RAGSequence")
    
    def answer(self):
        return RunnablePassthrough.assign(
            answer = RunnableLambda(lambda x: self.sequence())
        ).with_config(run_name="RAGAnswer")
        
    def run(self):
        return ((
            self.get_history_ctx()
            | self.context()
            | self.answer()
        ).assign(signature=lambda x: self.name)
        ).with_config(run_name=self.name)

    def get_ctx(self, user_input) -> str:
        relevant_docs = []
        # prendo i documenti che sono stati usati per rispondere alle domande precedenti
        follwoup_ctx = self.history.get_followup_ctx(self.followup_threshold)
        if follwoup_ctx:
            if type(follwoup_ctx[0]) is not Document:
                print(f"\33[1;31m[RAGChain]\33[0m: I documentisono di tipo {type(follwoup_ctx[0])}")
                raise Exception(TypeError)
            relevant_docs.extend(follwoup_ctx)
        # prendo i documenti che sono simili alla domanda dell'utente
        docs = self.retriever.invoke(user_input)
        if docs:
            relevant_docs.extend(docs)
        # rimuovo i documenti duplicati
        unique_docs = {}
        for doc in relevant_docs:
            try:
                unique_docs[doc.metadata.get('id', 0)] = doc
            except Exception as e:
                raise e
        sorted_docs = sorted(unique_docs.values(), key=lambda d: d.metadata.get('id', 0))
        if sorted_docs:
            return docs_to_string(sorted_docs)
        return ''

class ChainOfThoughts(HistoryAwareChain):
    """
    It's the chain which manages the entire conversation.
    To use it it is necessary to specify:
    - input
    """
    def __init__(self, llm: Runnable, handler: StdOutHandler | None, name: str, history: ChatHistory,
                 retriever: Retriever, retrieval_threshold: float, followup_threshold: float, distance_threshold: float):
        super().__init__(llm, handler, name, history)
        self.retriever = retriever
        
        self.retrieval_threshold = retrieval_threshold
        self.followup_threshold = followup_threshold
        self.distance_threshold = distance_threshold

        self.classification_chain = ClassificationChain(self.llm, self.handler, "ClassificationChain")
        self.conversational_chain = ConversationalChain(self.llm, self.handler, "ConversationalChain", self.history)
        self.summarization_chain = SummarizationChain(self.llm, self.handler, "SummarizationChain", self.history)
        self.RAG_chain = RAGChain(self.llm, self.handler, "RAGChain", self.history, self.retriever,
                                  self.retrieval_threshold, self.followup_threshold, self.distance_threshold)
        print("\33[1;34m[ChainOfThoughts]\33[0m: Chain inizializzata")

    def branch(self):
        return RunnableBranch(
            (RunnableLambda(lambda x: x.get('type') == 'summary'),
                self.summarization_chain.run()
            ),
            (RunnableLambda(lambda x: x.get('type') == 'document'),
                self.RAG_chain.run()
            ),
            (RunnableLambda(lambda x: x.get('type') == 'conversational'),
                self.conversational_chain.run()
            ),
            self.conversational_chain.run()
        ).with_config(run_name="ChainOfThoughtsBranch")
    
    def classify(self):
        return (
            RunnablePassthrough.assign(
                type = RunnableLambda(lambda x: self.classification_chain.run())
            )
            | RunnableLambda(lambda x: self.extract_type(x))
        ).with_config(run_name="ChainOfThoughtsClassification")
    
    def extract_type(self, inp: dict):
        old_type = inp.get('type', {})
        new_type = old_type.get('type', 'conversational')
        inp['type'] = new_type
        return inp
    
    def run(self):
        return ((
            self.classify() | self.branch()
        ).assign(signature=lambda x: self.name)
        ).with_config(run_name=self.name)