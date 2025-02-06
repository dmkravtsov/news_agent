import os
import json
import logging
import asyncio
from typing import List, Dict
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from pydantic import Field, PrivateAttr
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from data_models import NewsItem
from base_agent import BaseAgent
from dotenv import load_dotenv

os.environ["LOGFIRE_IGNORE_NO_CONFIG"] = "1"

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

load_dotenv(override=True)

class WriterAgent(BaseAgent):
    """Agent that analyzes news items and updates FAISS database for future context retrieval."""
    
    name: str = "WriterAgent"
    vector_db_path: str = Field(..., description="Path to FAISS vector database")
    translate_to_russian: bool = Field(default=False, description="Whether to translate output to Russian")
    _embedding_model: HuggingFaceEmbeddings = PrivateAttr()
    _vector_db: FAISS = PrivateAttr()
    _agent: Agent = PrivateAttr()

    def __init__(self, **data):
        """Initialize agent using Pydantic validation."""
        super().__init__(**data)  # Ensures Pydantic handles field validation
        
        # Initialize embedding model
        object.__setattr__(self, "_embedding_model", HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        ))

        # Initialize or load FAISS vector DB
        try:
            # Try to load existing DB
            object.__setattr__(self, "_vector_db", FAISS.load_local(
                self.vector_db_path, 
                self._embedding_model,
                allow_dangerous_deserialization=True
            ))
            self.log("✅ Loaded existing FAISS database")
        except Exception as e:
            # Create new DB if doesn't exist
            self.log(f"⚠ No existing FAISS database found, creating new one")
            object.__setattr__(self, "_vector_db", FAISS.from_texts(
                ["initial empty database"], 
                self._embedding_model
            ))
            self._vector_db.save_local(self.vector_db_path)
        
        # Initialize pydantic_ai agent
        object.__setattr__(self, "_agent", Agent(
            model=OpenAIModel(
                model_name="gpt-4",
                api_key=os.getenv("OPENAI_API_KEY")
            ),
            result_type=List[NewsItem],
            retries=3,
            system_prompt=f"""You are an expert news analyst specializing in extracting structured insights from news articles.
Your task is to analyze news content and provide detailed, accurate analysis{' in Russian' if self.translate_to_russian else ''}.

For each news item, analyze the content and return a NewsItem with these fields:
- description: Clear factual summary of the main event
- tags: List of key entities, topics, and themes (people, organizations, locations, key concepts)
- category: Specific category (Politics, Economics, Technology, Society, etc.)
- language: Language code ({"ru" if self.translate_to_russian else "en"})
- sentiment_score: Score from -0.0 (negative) to 1.0 (positive)
- future_trend: Specific likely consequences with timeframes
- precursors: Key events that led to this situation
- analytics: Data-driven insights and implications

Guidelines for analysis:
1. Sentiment Analysis:
   - Use -0.0 for extremely negative (disasters, deaths)
   - Use 0.5 for neutral news
   - Use 1.0 for extremely positive developments
2. Future Trends:
   - Include specific timeframes
   - Base on historical patterns
3. Precursors:
   - Focus on direct causes
   - Include dates when possible
4. Analytics:
   - Provide data-driven insights
   - Consider broader implications

{'''IMPORTANT: All text fields must be in Russian, including description, tags, category, future_trend, precursors, and analytics.
Keep names of people, organizations, and places in their original form but transliterated to Russian.''' if self.translate_to_russian else ''}

Analyze the provided news text and context, then return a structured NewsItem."""
        ))
        
        self.log(f"✅ WriterAgent initialized" + (" with Russian translation" if self.translate_to_russian else ""))

    async def process(self, news_items: List[Dict[str, str]]) -> List[NewsItem]:
        """Process news items and generate structured analysis"""
        if not news_items:
            self.log("⚠ No news items to process")
            return []

        self.log(f"🔄 Starting news analysis of {len(news_items)} items...")
        processed_news = []
        texts_to_add = []

        # Process each news item
        for i, item in enumerate(news_items, 1):
            self.log(f"📝 Processing item {i}/{len(news_items)}...")
            self.log(f"Content preview: {item['full_text'][:200]}...")  # Log preview of content
            
            attempts = 0
            while attempts < 3:
                try:
                    self.log(f"Attempt {attempts + 1} to process item {i}")
                    response = await self._agent.run(
                        user_prompt=f"""Analyze this news item:

Content: {item['full_text']}
Historical Context: {item.get('context', '')}
Date: {item.get('date', '')}

Return a structured NewsItem with all required fields."""
                    )
                    
                    if response and response.data:
                        # Log successful processing with key metadata
                        news_item = response.data[0]  # Get first item from response
                        self.log(f"""✅ Successfully processed item {i}:
                            - Category: {news_item.category}
                            - Tags: {', '.join(news_item.tags[:5])}{'...' if len(news_item.tags) > 5 else ''}
                            - Sentiment: {news_item.sentiment_score:.2f}""")
                        
                        processed_news.extend(response.data)
                        texts_to_add.append(item['full_text'])
                        break

                    self.log(f"⚠ No data returned for item {i}, attempt {attempts+1}/3")
                    attempts += 1
                except Exception as e:
                    self.log(f"❌ Error processing item {i}: {str(e)}", level="ERROR")
                    self.log(f"Error details: {type(e).__name__}", level="ERROR")
                    attempts += 1
                    await asyncio.sleep(2)

        # Update FAISS database
        if texts_to_add:
            try:
                original_size = len(self._vector_db.docstore._dict)
                self._vector_db.add_texts(texts_to_add)
                new_size = len(self._vector_db.docstore._dict)
                self._vector_db.save_local(self.vector_db_path)
                self.log(f"✅ Added {len(texts_to_add)} new items to FAISS database (Size: {original_size} → {new_size})")
            except Exception as e:
                self.log(f"❌ Error updating FAISS database: {str(e)}", level="ERROR")
                self.log(f"Error details: {type(e).__name__}", level="ERROR")

        # Log summary of processed items
        if processed_news:
            categories = {}
            avg_sentiment = 0
            for item in processed_news:
                categories[item.category] = categories.get(item.category, 0) + 1
                avg_sentiment += item.sentiment_score
            avg_sentiment /= len(processed_news)
            
            self.log(f"""✅ News processing summary:
                - Total items processed: {len(processed_news)}
                - Categories distribution: {categories}
                - Average sentiment: {avg_sentiment:.2f}""")
        
        return processed_news
