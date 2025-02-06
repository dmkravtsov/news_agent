from typing import List, Optional
from pydantic import BaseModel, Field

class NewsItem(BaseModel):
    """News item data model with structured analysis"""
    description: str = Field(..., description="A concise description of the news event")
    tags: List[str] = Field(default_factory=list, description="List of relevant tags, entities, and keywords maximum 3 items")
    category: str = Field(..., description="News category (e.g., Politics, Technology, Society)")
    language: str = Field(default="en", description="Language of the news item")
    sentiment_score: float = Field(..., ge=-0.0, le=1.0, description="Sentiment score from -0.0 (negative) to 1.0 (positive)")
    future_trend: str = Field(..., description="Predicted future developments and implications in 1-2 sentences maximum")
    precursors: str = Field(..., description="Events and context leading to this news in 1-2 sentences maximum")
    analytics: str = Field(..., description="Brief analysis of the news and its significance in 1-2 sentences maximum")

    def __str__(self):
        return (
            f"description='{self.description}' "
            f"tags={self.tags} "
            f"category='{self.category}' "
            f"language='{self.language}' "
            f"sentiment_score={self.sentiment_score} "
            f"future_trend='{self.future_trend}' "
            f"precursors='{self.precursors}' "
            f"analytics='{self.analytics}'"
        )


class NewsDigest(BaseModel):
    """News digest data model for publishing"""
    date_generated: str = Field(..., description="Date when the digest was generated")
    region: Optional[str] = Field(None, description="Optional region the digest covers") 
    news_items: List[NewsItem] = Field(default_factory=list, description="List of news items in the digest")
    summary: Optional[str] = Field(None, description="Optional overall summary of the digest")

    def __str__(self):
        """String representation of the digest"""
        parts = [
            f"News Digest ({self.date_generated})",
            f"Region: {self.region if self.region else 'Global'}",
            f"Number of news items: {len(self.news_items)}"
        ]
        if self.summary:
            parts.append(f"Summary: {self.summary}")
        return "\n".join(parts)
