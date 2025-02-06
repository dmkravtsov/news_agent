from typing import List, Dict, Optional
from pydantic import Field, PrivateAttr
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from base_agent import BaseAgent
from dotenv import load_dotenv
import os

load_dotenv()

class MCPService:
    """Model Context Protocol service for managing context windows and vector similarity search"""
    
    def __init__(self, embedding_model, vector_db):
        self.embedding_model = embedding_model
        self.vector_db = vector_db

    def preprocess_query(self, query: str) -> str:
        """Preprocess query text"""
        return query.lower().strip()

    def calculate_relevance_score(self, query_words: set, content: str) -> float:
        """Calculate relevance score based on word overlap"""
        content_words = set(content.lower().split())
        overlap = len(query_words & content_words)
        return overlap / len(content_words) if content_words else 0

    def is_same_news(self, query_words: set, content: str) -> bool:
        """Check if content is too similar to query"""
        content_words = set(content.lower().split())
        if not content_words:
            return False
        
        # Calculate Jaccard similarity
        intersection = len(query_words & content_words)
        union = len(query_words | content_words)
        jaccard_sim = intersection / union if union > 0 else 0
        
        # If Jaccard similarity is too high, likely same news
        return jaccard_sim > 0.6

    def get_relevant_context(self, query: str, top_k: int = 5) -> List[str]:
        """Get relevant context using vector similarity search with filtering"""
        # Preprocess once and create word set once
        query = self.preprocess_query(query)
        query_words = set(query.split())
        
        # Get documents using MMR for diversity with optimized parameters
        retriever = self.vector_db.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": top_k,
                "fetch_k": top_k * 2,
                "lambda_mult": 0.7
            }
        )
        
        docs = retriever.get_relevant_documents(query)
        
        # Filter and process results
        relevant_docs = []
        for doc in docs:
            content = doc.page_content.strip()
            # Skip if content is too short, too long, or same as query
            if len(content) < 50 or len(content) > 1000 or self.is_same_news(query_words, content):
                continue
            relevant_docs.append(content)
        
        return relevant_docs[:top_k] if relevant_docs else []

class ContextMakerAgent(BaseAgent):
    """Agent that searches for relevant news context in FAISS vector database
    and enriches news data with contextual information using LangChain and GPT-4."""
    
    name: str = "ContextMakerAgent"
    vector_db_path: str = Field(..., description="Path to FAISS vector database")
    _embedding_model: HuggingFaceEmbeddings = PrivateAttr()
    _vector_db: FAISS = PrivateAttr()
    _llm: ChatOpenAI = PrivateAttr()
    _chain: LLMChain = PrivateAttr()
    _mcp_service: MCPService = PrivateAttr()

    def __init__(self, **data):
        """Initialize agent using Pydantic validation."""
        super().__init__(**data)  # Ensures Pydantic handles field validation
        
        # Initialize embedding model
        object.__setattr__(self, "_embedding_model", HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        ))

        # Load FAISS vector DB
        try:
            object.__setattr__(self, "_vector_db", FAISS.load_local(
                self.vector_db_path, 
                self._embedding_model,
                allow_dangerous_deserialization=True
            ))
            self.log("✅ Loaded FAISS database")
        except Exception as e:
            self.log(f"❌ Failed to load FAISS database: {str(e)}", level="ERROR")
            raise  # Re-raise as we can't proceed without the database

        # Initialize MCP service
        object.__setattr__(self, "_mcp_service", MCPService(
            self._embedding_model,
            self._vector_db
        ))

        # Initialize LLM
        object.__setattr__(self, "_llm", ChatOpenAI(
            model_name="gpt-4",
            openai_api_key=os.getenv("OPENAI_API_KEY")
        ))

        # Initialize prompt template
        template = """
        You are analyzing news articles and their context.
        
        Main News Article:
        {query}
        
        Related News Articles for Context:
        {context}
        
        Task:
        1. Analyze how the related articles provide additional context to the main news
        2. Extract only the most relevant information that complements the main news
        3. If no relevant context is found, respond with an empty string
        4. Keep the response VERY concise - no more than 2 sentences
        5. Focus only on factual connections, avoid speculation
        
        Response:
        """
        
        object.__setattr__(self, "_prompt_template", PromptTemplate(
            input_variables=["query", "context"],
            template=template
        ))

        # Initialize LangChain
        object.__setattr__(self, "_chain", LLMChain(
            llm=self._llm,
            prompt=self._prompt_template
        ))
        
        self.log("✅ ContextMakerAgent initialized")

    async def process(self, news_items: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Process method required by BaseAgent. Delegates to add_context."""
        return await self.add_context(news_items)

    async def add_context(self, news_items: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        Process each news item sequentially, enriching it with relevant context from the database.
        """
        if not news_items:
            self.log("⚠ No news items to process")
            return []

        self.log("🔄 Starting context enrichment...")
        enriched_news = []
        processed_count = 0

        for item in news_items:
            try:
                # Find similar documents
                self.log(f"🔍 Searching for context in FAISS database for text: {item['full_text'][:100]}...")
                similar_docs = self._vector_db.similarity_search(
                    item['full_text'],
                    k=3
                )
                self.log(f"✅ Found {len(similar_docs)} similar documents")

                # Extract context from similar documents
                context_texts = [doc.page_content for doc in similar_docs]
                if context_texts:
                    # Generate context summary using LLM
                    context_summary = await self._chain.arun(
                        query=item['full_text'],
                        context="\n".join(context_texts)
                    )
                    
                    # Add context if meaningful
                    if context_summary and context_summary.strip():
                        item['context'] = context_summary.strip()
                        processed_count += 1
                        self.log(f"✅ Added context: {context_summary[:100]}...")
                
                enriched_news.append(item)
                
            except Exception as e:
                self.log(f"❌ Error processing item: {str(e)}", level="ERROR")
                enriched_news.append(item)  # Add original item without context

        self.log(f"✅ Added {processed_count} pieces of context to {len(news_items)} news items")
        return enriched_news
