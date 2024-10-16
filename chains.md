```mermaid
classDiagram
    class ChainInterface {
        <<interface>>
        
        + run()
        + invoke()
        + stream()
    }

    class Chain {
        + llm: LLM
        + handler: Handler
        + name: str

        + Chain() Chain
        + run() Runnable
        + invoke() dict
        + stream() dict
        + fill_prompt() Runnable
    }

    class HistoryAwareChain {
        + history: ChatHistory
        + num_messages: int

        + HistoryAwareChain() HistoryAwareChain
        + get_history_ctx() str
        + history_chain() Runnable
    }
    
    class ConversationalChain {
        + ConversationalChain() ConversationalChain
        + sequence() Runnable
        + answer() Runnable
        + run() Runnable
    }

    class ClassificationChain {
        + ClassificationChain() ClassificationChain
        + fill_prompt() Runnable
        + sequence() Runnable
        + classify() Runnable
        + run() Runnable
    }

    class SummarizationChain {
        + SummarizationChain() SummarizationChain
        + fill_prompt() Runnable
        + sequence() Runnable
        + answer() Runnable
        + run() Runnable
    }

    class RAGChain {
        + retriever: Retriever
        + retrieval_threshold: float
        + follow_up_threshold: float
        + embedding_threshold: float

        + RAGChain() RAGChain
        + fill_prompt() Runnable
        + context() Runnable
        + sequence() Runnable
        + answer() Runnable
        + get_ctx() str
        + run() Runnable
    }

    class ChainOfThoughts {
        + retriever: Retriever
        + retrieval_threshold: float
        + follow_up_threshold: float
        + embedding_threshold: float
        + rag_chain: RAGChain
        + summarization_chain: SummarizationChain
        + classification_chain: ClassificationChain

        + ChainOfThoughts() ChainOfThoughts
        + branch() Runnable
        + run() Runnable
    }

    ChainInterface <|.. Chain

    Chain <|-- HistoryAwareChain
    Chain <|-- ClassificationChain

    HistoryAwareChain <|-- ConversationalChain
    HistoryAwareChain <|-- SummarizationChain
    HistoryAwareChain <|-- RAGChain
    HistoryAwareChain <|-- ChainOfThoughts

    ChainOfThoughts o-- ConversationalChain
    ChainOfThoughts o-- RAGChain
    ChainOfThoughts o-- SummarizationChain
    ChainOfThoughts o-- ClassificationChain
```