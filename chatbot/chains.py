from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.documents.base import Document
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import (
    RunnableBranch,
    RunnableLambda,
    RunnableSequence,
    RunnablePassthrough,
    Runnable
)

from retriever import Retriever
from utilities import ChatHistory, StdOutHandler
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

class Chain(ChainInterface):
    def __init__(self, llm: Runnable, handler: StdOutHandler | None):
        self.llm = llm
        self.handler = handler
    
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

    def stream(self, input, containers=None):
        try:
            if self.handler:
                self.handler.start(containers)
            response = {}
            out = self.run().stream(input)
            for token in out:
                if self.handler:
                    self.handler.on_new_token(token)
                else: # ! DEBUG
                    if token.get('answer'):
                        print(token.get('answer'), sep='', end='', flush=True)
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
        return ChatPromptTemplate.from_messages(
            [
                ("system", system_template),
                ("human", "{input}")
            ]
        ).with_config(run_name="ChatPromptTemplate")

class HistoryAwareChain(Chain):
    def __init__(self, llm: Runnable, handler: StdOutHandler | None, history: ChatHistory, num_messages: int):
        super().__init__(llm, handler)
        self.history = history
        self.num_messages = num_messages
    
    def get_history_ctx(self, n):
        history_ctx = ["CHAT HISTORY:"]
        ctx = self.history.get_last_messages(n)
        if ctx:
            for msg in ctx:
                if isinstance(msg, AIMessage):
                    m = "AI: " + msg.content
                elif isinstance(msg, HumanMessage):
                    m = "HUMAN: " + msg.content
                history_ctx.append(m)
            return "\n".join(history_ctx)
        else:
            return ""
        
    def history_chain(self):
        return RunnablePassthrough.assign(
            history_ctx = RunnableLambda(lambda x: self.get_history_ctx(self.num_messages))
        ).with_config(run_name="HistoryCTX")

CONVERSATION_TEMPLATE = """ \
Tu sei un assistente che risponde alle domande relative all'Aeronautica Militare Italiana. \
Rispondi sempre in ITALIANO, e solo alle domande che possono avere a che fare con l'aeronautica:
ESEMPI: piloti, aerei, carriere, accademia, corsi, concorsi, bandi... \
NON rispondere ad una domanda con un'altra domanda. \
Se la domanda NON è inerente al contesto, rispondi con "Non so rispondere a questa domanda". \
Dialoga con l'utente e rispondi alle sue domande. \
Se ti dovessero chiedere il tuo nome, tu ti chiami Turi.
Se l'utente ti ringrazia, rispondi con "Prego" o "Non c'è di che" e renditi sempre disponibile. \
Cerca di rispondere in modo adeguato alla conversazione.

{history_ctx}
"""

class ConversationalChain(HistoryAwareChain):
    def __init__(self, llm: Runnable, handler: StdOutHandler | None, history: ChatHistory, num_messages: int):
        super().__init__(llm, handler, history, num_messages)
        self.name = 'ConversationalChain'
        print("\33[1;36m[ConversationalChain]\33[0m: Chain inizializzata")
    
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
            RunnableLambda(lambda x: self.history_chain())
            | self.answer()
        ).assign(signature=lambda x: self.name)
        ).with_config(run_name=self.name)

DOCUMENT_TEMPLATE = """ \
Tu sei un assistente che risponde alle domande relative all'Aeronautica Militare Italiana. \
Rispondi sempre in ITALIANO, e solo alle domande che possono avere a che fare con l'aeronautica:
ESEMPI: piloti, aerei, carriere, accademia, corsi, concorsi, bandi... \
NON rispondere ad una domanda con un'altra domanda. \
L'utente NON deve sapere che stai rispondendo grazie ai seguenti documenti. \

{history_ctx}

CONTESTO:
{context}
"""

class DocumentChain(HistoryAwareChain):
    """
    It's a RAG Chain with context passed as parameter.
    To use it it is necessary to specify:
    - input
    - context
    """
    def __init__(self, llm: Runnable, handler: StdOutHandler | None, history: ChatHistory, num_messages: int):
        super().__init__(llm, handler, history, num_messages)
        self.name = 'DocumentChain'
        print("\33[1;36m[DocumentChain]\33[0m: Chain inizializzata")
        
    def sequence(self):
        return RunnableSequence(
            self.fill_prompt(DOCUMENT_TEMPLATE),
            self.llm
        ).with_config(run_name="DocumentSequence")
    
    def answer(self):
        return RunnablePassthrough.assign(
            answer = RunnableLambda(lambda x: self.sequence())
        ).with_config(run_name="DocumentAnswer")
    
    def run(self):
        return ((
            RunnableLambda(lambda x: self.history_chain())
            | self.answer()
        ).assign(signature=lambda x: self.name)
        ).with_config(run_name=self.name)

class DefaultChain(Chain):
    """
    It's the chain which manages the DocumentChain and the ConversationalChain.
    To use it it is necessary to specify:
    - input
    """
    def __init__(self, llm: Runnable, handler: StdOutHandler | None,
                 retriever: Retriever, threshold: float, document_chain: Runnable, conversational_chain: Runnable):
        super().__init__(llm, handler)
        self.name = "DefaultChain"
        self.retriever = retriever
        self.threshold = threshold

        self.document_chain = document_chain
        self.conversational_chain = conversational_chain
        print("\33[1;36m[DefaultChain]\33[0m: Chain inizializzata")

    def context(self):
        return RunnablePassthrough.assign(
            context = RunnableLambda(lambda x: self.retriever.retrieve(x.get('input'), self.threshold))
        ).with_config(run_name="DefaultContext")
    
    def branch(self):
        return RunnableBranch(
            (RunnableLambda(lambda x: x.get('context') != []),
                self.document_chain.run()
            ),
            self.conversational_chain.run()
        ).with_config(run_name="DefaultBranch")
    
    def run(self):
        return ((
            RunnableLambda(lambda x: self.context())
            | self.branch()
        ).assign(signature=lambda x: self.name)
        ).with_config(run_name=self.name)
    
CLASSIFICATION_TEMPLATE = """
Stai parlando con un utente e devi classificare le sue domande in quattro categorie: "summary", "followup", "document" e "conversational".

- Le domande "summary" richiedono un riassunto delle informazioni precedenti. Esempi: "Puoi fare un riassunto?", "Riassumi ciò di cui abbiamo parlato."
- Le domande "followup" sono domande che si basano sul contesto degli ultimi messaggi. Esempi: "Dimmi di più", "Approfondisci questo punto, "Fammi capire meglio", "Perché è così?", "Continua".
- Le domande "document" sono domande che chiedono di argomenti specifici basati su documenti forniti. Esempi: "Qual è la capitale della Francia?", "Come si usa un saldatore?."
- Le domande "conversational" sono domande che NON sono basate su documenti forniti. Esempi: "Ciao", "Che cosa sai fare?", "Come ti chiami?", "Chi ti ha creato?".

DOMANDA:
{input}

Rispondi con un JSON che indica il tipo di domanda.
Esempio: {{"type": "summary"}} o {{"type": "followup"}} o {{"type": "document"}} o {{"type": "conversational"}}
"""

class ClassificationChain(Chain):
    """
    It's the chain which classifies the user's questions in three categories: "summary", "followup", and "standard".
    To use it it is necessary to specify:
    - input
    """
    def __init__(self, llm: Runnable, handler: StdOutHandler | None):
        super().__init__(llm, handler)
        self.name = "ClassificationChain"
        print("\33[1;36m[ClassificationChain]\33[0m: Chain inizializzata")
    
    #* Override
    def fill_prompt(self):
        return ChatPromptTemplate.from_template(CLASSIFICATION_TEMPLATE).with_config(run_name="ChatPromptTemplate")
    
    def sequence(self):
        return RunnableSequence(
            self.fill_prompt(),
            self.llm,
            JsonOutputParser()
        ).with_config(run_name="ClassificationSequence")
    
    def classify(self):
        return RunnablePassthrough.assign(
            type = RunnableLambda(lambda x:
                self.sequence().invoke({"input": x.get('input')}).get('type', 'conversational'))
        ).with_config(run_name="ClassifyInput")
        
    def run(self):
        return ((
            RunnableLambda(lambda x: self.classify())
        ).assign(signature=lambda x: self.name)
        ).with_config(run_name=self.name)

SUMMARIZATION_TEMPLATE = """
Stai parlando con un utente e devi fare un riassunto delle informazioni di cui avete discusso. \
L'utente ti ha chiesto di farlo con questa domanda: "{input}".
NON ripetere la domanda dell'utente. \
Rispondi in ITALIANO (o nella lingua della domanda) rispettando la richiesta dell'utente e utilizzando le informazioni seguenti. \

{history_ctx}
"""

class SummarizationChain(HistoryAwareChain):
    """
    It's the chain which summarizes the information discussed with the user.
    To use it it is necessary to specify:
    - input
    """
    def __init__(self, llm: Runnable, handler: StdOutHandler | None, history: ChatHistory, num_messages: int):
        super().__init__(llm, handler, history, num_messages)
        self.name = "SummarizationChain"
        print("\33[1;36m[SummarizationChain]\33[0m: Chain inizializzata")
    
    #* Override
    def fill_prompt(self):
        return PromptTemplate.from_template(SUMMARIZATION_TEMPLATE).with_config(run_name="ChatPromptTemplate")
        
    def sequence(self):
        return RunnableSequence(
            self.fill_prompt(),
            self.llm
        ).with_config(run_name="SummarizationSequence")
        
    def answer(self):
        return RunnablePassthrough.assign(
            answer = RunnableLambda(lambda x: self.sequence())
        ).with_config(run_name="SummarizationAnswer")
    
    def run(self):
        return ((
            RunnableLambda(lambda x: self.history_chain())
            | self.answer()
        ).assign(signature=lambda x: self.name)
        ).with_config(run_name=self.name)

FOLLOWUP_TEMPLATE = """
Stai parlando con un utente e devi approfondire un punto specifico. \
L'utente ti ha chiesto di farlo con questa domanda: "{input}". \

{history_ctx}
NON ripetere la domanda dell'utente e NON dire che sai che lui vuole un approfondimento. \
Rispondi in ITALIANO (o nella lingua della domanda) rispettando la richiesta dell'utente e utilizzando le informazioni seguenti. \
CONTESTO:

{context}
"""

class FollowupChain(HistoryAwareChain):
    """
    It's the chain which manages the followup questions.
    To use it it is necessary to specify:
    - input
    """
    def __init__(self, llm: Runnable, handler: StdOutHandler | None, history: ChatHistory, num_messages: int,
                 retriever: Retriever, retrieval_threshold: float, followup_threshold: float, embedding_threshold: float):
        super().__init__(llm, handler, history, num_messages)
        self.name = "FollowupChain"
        self.retriever = retriever
        
        self.retrieval_threshold = retrieval_threshold
        self.followup_threshold = followup_threshold
        self.embedding_threshold = embedding_threshold
        print("\33[1;36m[FollowupChain]\33[0m: Chain inizializzata")
    
    #* Override
    def fill_prompt(self):
        return PromptTemplate.from_template(FOLLOWUP_TEMPLATE).with_config(run_name="ChatPromptTemplate")
    
    def context(self):
        return RunnablePassthrough.assign(
            context = RunnableLambda(lambda x: self.get_ctx(x.get('input')))
        ).with_config(run_name="FollowupContext")
    
    def sequence(self):
        return RunnableSequence(
            self.fill_prompt(),
            self.llm
        ).with_config(run_name="FollowupSequence")
    
    def answer(self):
        return RunnablePassthrough.assign(
            answer = RunnableLambda(lambda x: self.sequence())
        ).with_config(run_name="FollowupAnswer")
        
    def run(self):
        return ((
            RunnableLambda(lambda x: self.context())
            | RunnableLambda(lambda x: self.history_chain())
            | self.answer()
        ).assign(signature=lambda x: self.name)
        ).with_config(run_name=self.name)
    
    def get_ctx(self, user_input) -> list[Document]:
        relevant_docs = []
        # prendo i documenti che sono stati usati per rispondere alle domande precedenti
        folloup_ctx = self.history.get_followup_ctx(user_input, self.followup_threshold)
        if folloup_ctx:
            relevant_docs.extend(folloup_ctx)
        # prendo i documenti che sono simili alla domanda dell'utente
        docs = self.retriever.retrieve(user_input, self.retrieval_threshold)
        relevant_docs.extend(docs)
        # prendo i documenti simili ai documenti trovati
        augmented_ctx = []
        for doc in relevant_docs:
            similar_docs = self.retriever.find_similar(doc, self.embedding_threshold)
            if similar_docs:
                augmented_ctx.extend(similar_docs)
        # se ci sono documenti simili, li aggiungo al contesto
        if augmented_ctx:
            relevant_docs.extend(augmented_ctx)
        # rimuovo i documenti duplicati
        unique_docs = {}
        for doc in relevant_docs:
            try:
                unique_docs[doc.metadata['id']] = doc
            except Exception as e:
                raise e
        return list(unique_docs.values())

class ChainOfThoughts(HistoryAwareChain):
    """
    It's the chain which manages the entire conversation.
    To use it it is necessary to specify:
    - input
    """
    def __init__(self, llm: Runnable, handler: StdOutHandler | None, history: ChatHistory, num_messages: int,
                 retriever: Retriever, retrieval_threshold: float, followup_threshold: float, embedding_threshold: float):
        super().__init__(llm, handler, history, num_messages)
        self.name = "ChainOfThoughts"
        self.retriever = retriever
        
        self.retrieval_threshold = retrieval_threshold
        self.followup_threshold = followup_threshold
        self.embedding_threshold = embedding_threshold

        self.classification_chain = ClassificationChain(self.llm, self.handler)
        self.document_chain = DocumentChain(self.llm, self.handler, self.history, self.num_messages)
        self.conversational_chain = ConversationalChain(self.llm, self.handler, self.history, self.num_messages)
        self.default_chain = DefaultChain(self.llm, self.handler, self.retriever, self.retrieval_threshold,
                                          self.document_chain, self.conversational_chain)
        self.summarization_chain = SummarizationChain(self.llm, self.handler, self.history, self.num_messages)
        self.followup_chain = FollowupChain(self.llm, self.handler, self.history, self.num_messages,
                                            self.retriever, self.retrieval_threshold, self.followup_threshold, self.embedding_threshold)
        print("\33[1;36m[ChainOfThoughts]\33[0m: Chain inizializzata")

    def branch(self):
        return RunnableBranch(
            (RunnableLambda(lambda x: x.get('type') == 'summary'),
                self.summarization_chain.run()
            ),
            (RunnableLambda(lambda x: x.get('type') == 'followup'),
                self.followup_chain.run()
            ),
            (RunnableLambda(lambda x: x.get('type') == 'conversational'),
                self.conversational_chain.run()
            ),
            (RunnableLambda(lambda x: x.get('type') == 'document'),
                self.default_chain.run()
            ),
            self.default_chain.run()
        ).with_config(run_name="ChainOfThoughtsBranch")

    def branch_no_history(self):
        return RunnableBranch(
            (RunnableLambda(lambda x: x.get('type') == 'document'),
                self.default_chain.run()
            ),
            (RunnableLambda(lambda x: x.get('type') == 'conversational'),
                self.conversational_chain.run()
            ),
            self.default_chain.run()
        ).with_config(run_name="ChainOfThoughtsBranch_1EX")
    
    def isFirst(self):
        return RunnableBranch(
            (RunnableLambda(lambda x: self.history.messages != []),
                self.branch()
            ),
            self.branch_no_history()
        ).with_config(run_name="ChainOfThoughtsIsFirst")
    
    def run(self):
        return ((
            self.classification_chain.run()
            | self.isFirst()
        ).assign(signature=lambda x: self.name)
        ).with_config(run_name=self.name)