from pydantic import Field, PrivateAttr
from sentence_transformers import SentenceTransformer
from base_agent import BaseAgent
from typing import List, Dict, Set
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import spacy

class ManagerAgent(BaseAgent):
    """
    Agent for processing and grouping news based on semantic similarity and named entities (PERSON, ORG).
    """
    name: str = "ManagerAgent"
    similarity_threshold: float = Field(..., description="Threshold for merging similar news")
    _model: SentenceTransformer = PrivateAttr()
    _nlp: spacy.Language = PrivateAttr()

    def __init__(self, similarity_threshold: float = 0.3):
        """Initialize agent using Pydantic validation."""
        super().__init__(similarity_threshold=similarity_threshold)  # Ensures Pydantic handles field validation
        self._model = SentenceTransformer("all-MiniLM-L6-v2")
        self._nlp = spacy.load("en_core_web_sm")  # Load spaCy model for NER
        self.log("✅ ManagerAgent initialized")

    def extract_named_entities(self, text: str) -> Set[str]:
        """
        Extracts named entities (NER) of type PERSON or ORG from the text.
        """
        doc = self._nlp(text)
        entities = {ent.text.lower() for ent in doc.ents if ent.label_ in {"PERSON", "ORG"}}  # Only PERSON & ORG
        return entities

    async def process(self, news_items: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        Processes and groups news based on similarity and common named entities (NER).
        Outputs grouped news in the same format as CollectorAgent: {"full_text": "...", "date": "..."}.
        """
        print("\n[ManagerAgent] ManagerAgent initialized.")

        if not news_items:
            await self.log("No news data received.", level="WARNING")
            print("[ManagerAgent] No news data received.")
            return []

        news_texts = [item["full_text"] for item in news_items]
        news_dates = [item["date"] for item in news_items]

        embeddings = self._model.encode(news_texts, convert_to_tensor=True)
        similarity_matrix = cosine_similarity(embeddings)

        news_entities = {i: self.extract_named_entities(text) for i, text in enumerate(news_texts)}

        grouped_news = []
        visited = set()

        for i, item in enumerate(news_items):
            if i in visited:
                continue

            text = item["full_text"]
            date = item["date"]

            similar_texts = [text]
            dates = [date]
            visited.add(i)

            for j in range(i + 1, len(news_items)):
                if j in visited:
                    continue

                cosine_match = similarity_matrix[i][j] >= self.similarity_threshold
                common_entities = news_entities[i] & news_entities[j]  # Intersection of named entities
                ner_match = len(common_entities) > 0  # At least one common entity

                if cosine_match or ner_match:  # 🔥 Учитываем оба критерия!
                    similar_texts.append(news_items[j]["full_text"])
                    dates.append(news_items[j]["date"])
                    visited.add(j)

            max_date = max(dates)  # Выбираем самую позднюю дату объединённых новостей

            grouped_news.append({
                "full_text": " | ".join(set(similar_texts)),  # Объединяем схожие новости
                "date": max_date
            })

        print(f"[ManagerAgent] Grouped {len(grouped_news)} news clusters.\n")
        self.log(f"Grouped {len(grouped_news)} news clusters.")

        return grouped_news
