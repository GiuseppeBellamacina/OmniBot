from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.documents.base import Document
from langchain_core.runnables import (
    RunnableBranch,
    RunnableLambda,
    RunnableSequence,
    RunnablePassthrough,
    Runnable
)

from retriever import Retriever

class Chain():
    def __init__(self, llm, handler=None):
        self.llm = llm
        self.handler = handler
        self.chain = self.llm
    
    def invoke(self, input, containers=None):
        try:
            if self.handler:
                self.handler.start(containers)
            response = self.chain.invoke(input)
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
            out = self.chain.stream(input)
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
"""

class ConversationalChain(Chain):
    def __init__(self, llm, handler=None):
        super().__init__(llm, handler)
        self.name = 'ConversationalChain'
        print("\33[1;36m[ConversationalChain]\33[0m: Chain inizializzata")
        
        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", CONVERSATION_TEMPLATE),
                ("human", "{input}")
            ]
        ).with_config(run_name="ConversationalChainPrompt")
        
        self.sequence = RunnableSequence(
            self.prompt,
            self.llm
        ).with_config(run_name="ConversationalChainSequence")
        
        self.chain = (
            RunnablePassthrough.assign(
                answer=self.sequence
            ).with_config(run_name="ConversationalChainAnswer").assign(signature=lambda x: self.name)
        ).with_config(run_name=self.name)

DOCUMENT_TEMPLATE = """ \
Tu sei un assistente che risponde alle domande relative all'Aeronautica Militare Italiana. \
Rispondi sempre in ITALIANO, e solo alle domande che possono avere a che fare con l'aeronautica:
ESEMPI: piloti, aerei, carriere, accademia, corsi, concorsi, bandi... \
NON rispondere ad una domanda con un'altra domanda. \
L'utente NON deve sapere che stai rispondendo grazie ai seguenti documenti.
CONTESTO:

{context}"""

class DocumentChain(Chain):
    """
    It's a RAG Chain with context passed as parameter.
    To use it it is necessary to specify:
    - input
    - context
    """
    def __init__(self, llm, handler, retriever: Retriever, threshold: float):
        super().__init__(llm, handler)
        self.name = 'DocumentChain'
        self.retriever = retriever
        self.threshold = threshold
        
        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", DOCUMENT_TEMPLATE),
                ("human", "\nDOMANDA: {input}")
            ]
        ).with_config(run_name="DocumentChainPrompt")
        
        self.sequence = RunnableSequence(
            self.prompt,
            self.llm
        ).with_config(run_name="DocumentChainSequence")
        
        self.chain = (
            RunnablePassthrough.assign(
                answer=self.sequence
            ).with_config(run_name="DocumentChainAnswer").assign(signature=lambda x: self.name)
        ).with_config(run_name=self.name)
        
        print("\33[1;36m[DocumentChain]\33[0m: Chain inizializzata")

class DefaultChain(Chain):
    """
    It's the chain which manages the DocumentChain and the ConversationalChain.
    To use it it is necessary to specify:
    - input
    """
    def __init__(self, llm, handler, retriever: Retriever, threshold: float, document_chain: Runnable, conversational_chain: Runnable):
        super().__init__(llm, handler)
        self.name = "DefaultChain"
        self.retriever = retriever
        self.threshold = threshold

        self.document_chain = document_chain
        self.conversational_chain = conversational_chain

        self.retrieval = RunnablePassthrough.assign(
            context = RunnableLambda(lambda x: self.retriever.retrieve(x.get('input'), self.threshold))
        ).with_config(run_name="DefaultChainRetrieval")

        self.branch = RunnableBranch(
            (RunnableLambda(lambda x: x.get('context') != []),
                self.document_chain
            ),
            self.conversational_chain
        ).with_config(run_name="DefaultChainBranch")

        self.chain = (
            self.retrieval | self.branch
        ).with_config(run_name=self.name)

        print("\33[1;36m[DefaultChain]\33[0m: Chain inizializzata")
    

CLASSIFICATION_TEMPLATE = """
Stai parlando con un utente e devi classificare le sue domande in tre categorie: "summary", "followup", "document" e "conversational".

- Le domande "summary" richiedono un riassunto delle informazioni precedenti. Esempi: "Puoi fare un riassunto?", "Riassumi ciò di cui abbiamo parlato."
- Le domande "followup" sono domande che si basano sul contesto degli ultimi messaggi. Esempi: "Dimmi di più", "Approfondisci questo punto, "Fammi capire meglio", "Perché è così?", "Continua".
- Le domande "document" sono domande che non richiedono contesto precedente ma che chiedono di argomenti specifici basati su documenti forniti. Esempi: "Qual è la capitale della Francia?", "Come si usa un saldatore?."
- Le domande "conversational" sono domande che non richiedono contesto precedente e che non sono basate su documenti forniti. Esempi: "Ciao", "Che cosa sai fare?", "Come ti chiami?", "Chi ti ha creato?".

Domanda:
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
    def __init__(self, llm, handler):
        super().__init__(llm, handler)
        self.name = "ClassificationChain"
        
        self.prompt = ChatPromptTemplate.from_template(CLASSIFICATION_TEMPLATE).with_config(run_name="ClassificationChainPrompt")
        self.sequence = RunnableSequence(
            self.prompt,
            self.llm,
            JsonOutputParser()
        ).with_config(run_name="ClassificationChainSequence")
        
        self.chain = (
            RunnablePassthrough.assign(
                type=RunnableLambda(lambda x: self.sequence.invoke({"input": x.get('input')}).get('type', 'conversational'))
            ).with_config(run_name="ClassificationChainType")
        ).with_config(run_name=self.name)
        
        print("\33[1;36m[ClassificationChain]\33[0m: Chain inizializzata")

SUMMARIZATION_TEMPLATE = """
Stai parlando con un utente e devi fare un riassunto delle informazioni di cui avete discusso. \
L'utente ti ha chiesto di farlo con questa domanda: "{input}". \
NON ripetere la domanda dell'utente. \
Rispondi in ITALIANO (o nella lingua della domanda) rispettando la richiesta dell'utente e utilizzando le informazioni seguenti. \
CHAT HISTORY:

{chat_history}
"""

class SummarizationChain(Chain):
    """
    It's the chain which summarizes the information discussed with the user.
    To use it it is necessary to specify:
    - input
    """
    def __init__(self, llm, history, handler):
        super().__init__(llm, handler)
        self.name = "SummarizationChain"
        self.history = history
        
        self.prompt = PromptTemplate.from_template(SUMMARIZATION_TEMPLATE).with_config(run_name="SummarizationChainPrompt")
        self.sequence = RunnableSequence(
            self.prompt,
            self.llm
        ).with_config(run_name="SummarizationChainSequence")

        self.get_history_chain = (
            RunnablePassthrough.assign(
                chat_history = RunnableLambda(lambda x: self.history.get_all_messages())
            ).with_config(run_name="GetHistoryPassthrough")
        ).with_config(run_name="GetHistoryChain")
        
        self.chain = (
            RunnablePassthrough.assign(
                answer=self.get_history_chain | self.sequence
            ).with_config(run_name="SummarizationChainAnswer").assign(signature=lambda x: self.name)
        ).with_config(run_name=self.name)
        
        print("\33[1;36m[SummarizationChain]\33[0m: Chain inizializzata")

FOLLOWUP_TEMPLATE = """
Stai parlando con un utente e devi approfondire un punto specifico. \
L'utente ti ha chiesto di farlo con questa domanda: "{input}". \
NON ripetere la domanda dell'utente e NON dire che sai che lui vuole un approfondimento. \
Rispondi in ITALIANO (o nella lingua della domanda) rispettando la richiesta dell'utente e utilizzando le informazioni seguenti. \
CONTESTO:

{context}
"""

class FollowupChain(Chain):
    """
    It's the chain which manages the followup questions.
    To use it it is necessary to specify:
    - input
    """
    def __init__(self, llm, retriever: Retriever, threshold: float, history, handler):
        super().__init__(llm, handler)
        self.name = "FollowupChain"
        self.retriever = retriever
        self.threshold = threshold
        self.history = history
        
        self.prompt = PromptTemplate.from_template(FOLLOWUP_TEMPLATE).with_config(run_name="FollowupChainPrompt")
        
        self.retrieve_ctx = RunnablePassthrough.assign(
            context = RunnableLambda(lambda x: self.get_ctx(x.get('input')))
        ).with_config(run_name="FollowupContext")
        
        self.sequence = (
            RunnablePassthrough.assign(
                answer = self.prompt | self.llm
            ).with_config(run_name="FollowupAnswer").assign(signature=lambda x: self.name)
        ).with_config(run_name="FollowupSequence")
        
        self.chain = (
            RunnableLambda(lambda x: self.retrieve_ctx)
            | self.sequence
        ).with_config(run_name=self.name)
        
        print("\33[1;36m[FollowupChain]\33[0m: Chain inizializzata")
    
    def get_ctx(self, user_input) -> list[Document]:
        relevant_docs = []
        folloup_ctx = self.history.get_followup_ctx(user_input, 0.4)
        print("\33[1;36m[FollowupChain]\33[0m: Contesto recuperato")
        print(folloup_ctx)
        if folloup_ctx:
            relevant_docs.extend(folloup_ctx)
        docs = self.retriever.retrieve(user_input, self.threshold)
        print("\33[1;36m[FollowupChain]\33[0m: Risultati recuperati")
        print(docs)
        relevant_docs.extend(docs)
        unique_docs = {}
        for doc in relevant_docs:
            unique_docs[doc.metadata['id']] = doc
        return list(unique_docs.values())

class ChainOfThoughts(Chain):
    """
    It's the chain which manages the entire conversation.
    To use it it is necessary to specify:
    - input
    """
    def __init__(self, llm, retriever: Retriever, threshold: float, history, handler):
        super().__init__(llm, handler)
        self.name = "ChainOfThoughts"
        self.retriever = retriever
        self.threshold = threshold
        self.history = history
        self.handler = handler

        self.classification_chain = ClassificationChain(self.llm, self.handler).chain
        self.document_chain = DocumentChain(self.llm, self.handler, self.retriever, self.threshold).chain
        self.conversational_chain = ConversationalChain(self.llm, self.handler).chain
        self.default_chain = DefaultChain(self.llm, self.handler, self.retriever, self.threshold, self.document_chain, self.conversational_chain).chain
        self.summarization_chain = SummarizationChain(self.llm, self.history, self.handler).chain
        self.followup_chain = FollowupChain(self.llm, self.retriever, self.threshold, self.history, self.handler).chain

        self.branch = RunnableBranch(
            (RunnableLambda(lambda x: x.get('type') == 'summary'),
                self.summarization_chain
            ),
            (RunnableLambda(lambda x: x.get('type') == 'followup'),
                self.followup_chain
            ),
            (RunnableLambda(lambda x: x.get('type') == 'conversational'),
                self.conversational_chain
            ),
            (RunnableLambda(lambda x: x.get('type') == 'document'),
                self.default_chain
            ),
            self.default_chain
        ).with_config(run_name="ChainOfThoughtsBranch")

        self.firstMessageBranch = RunnableBranch(
            (RunnableLambda(lambda x: x.get('type') == 'document'),
                self.default_chain
            ),
            (RunnableLambda(lambda x: x.get('type') == 'conversational'),
                self.conversational_chain
            ),
            self.default_chain
        ).with_config(run_name="ChainOfThoughtsFirstBranch")

        self.isFirst = RunnableBranch(
            (RunnableLambda(lambda x: self.history.messages != []).with_config(run_name="HistoryCheck"),
                self.branch
            ),
            self.firstMessageBranch
        ).with_config(run_name="ChainOfThoughtsIsFirst")

        self.chain = (
            self.classification_chain | self.isFirst
        ).with_config(run_name=self.name)
        print("\33[1;36m[ChainOfThoughts]\33[0m: Chain inizializzata")