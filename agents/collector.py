import aiohttp
import feedparser
import re
from datetime import datetime, timedelta
from typing import List, Dict, Any
from base_agent import BaseAgent
from dateutil import parser
from pydantic import Field

# Possible date keys in RSS feeds
DATE_KEYS = ["dc:date", "published", "pubDate", "date", "updated", "created"]

class CollectorAgent(BaseAgent):
    """
    Agent for collecting news from RSS feeds.
    """
    name: str = "CollectorAgent"
    source_configs: List[Dict[str, Any]] = Field(default_factory=list, description="List of sources for RSS parsing")

    def __init__(self, **data):
        """Initialize agent using Pydantic validation."""
        super().__init__(**data)  # Ensures Pydantic handles field validation
        self.log("✅ CollectorAgent initialized.")

    async def process(self, source_configs: List[Dict[str, Any]] = None) -> List[Dict[str, str]]:
        """
        Process news collection from specified sources.
        :param source_configs: Optional list of source configurations. If not provided, uses the instance's source_configs.
        :return: List of news articles in the format {full_text, date}.
        """
        if source_configs is not None:
            self.source_configs = source_configs
            
        return await self.collect(self.source_configs)

    async def collect(self, source_configs: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """
        Collect news from specified sources.
        :param source_configs: List of source configurations
        :return: List of news articles in the format {full_text, date}.
        """
        self.source_configs = source_configs
        self.log("🔄 Starting news collection...")

        all_news = []
        for source in self.source_configs:
            url = source["url"]
            days_ago = source["days_ago"]
            start_date = self.get_start_date(days_ago)
            end_date = self.get_end_date()

            news_items = await self.fetch_rss_news(url, start_date, end_date)
            all_news.extend(news_items)

        self.log(f"✅ News processing completed. Collected {len(all_news)} news items.")
        return all_news

    async def fetch_rss_news(self, url: str, start_date: datetime, end_date: datetime) -> List[Dict[str, str]]:
        """
        Loads an RSS feed, filters news by date, and returns a list of news headlines.
        """
        headers = {"User-Agent": "Mozilla/5.0"}
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers) as response:
                    response.raise_for_status()
                    content = await response.read()
                    feed = feedparser.parse(content)
        except Exception as e:
            self.log(f"⚠ Error fetching RSS feed from {url}: {e}", level="WARNING")
            return []

        news_items = []
        for entry in feed.entries:
            # Find the date in possible keys
            final_date = None
            for key in DATE_KEYS:
                if key in entry:
                    try:
                        final_date = parser.parse(entry[key])
                        final_date = final_date.replace(tzinfo=None)  # Remove timezone
                        break
                    except Exception:
                        pass

            # If date exists, filter by range
            if final_date and (start_date <= final_date <= end_date):
                title = entry.get("title", "")
                description = entry.get("description", "")

                clean_title = re.sub(r"<.*?>", " ", title).strip()
                clean_description = re.sub(r"<.*?>", " ", description).strip()
                clean_description = re.sub(r"\s*Continue reading\.\.\.", "", clean_description)

                full_text = f"{clean_title}. {clean_description}".strip()

                news_items.append({
                    "full_text": full_text,
                    "date": final_date.isoformat()  # Convert date to ISO format
                })

        return news_items

    def get_start_date(self, days_ago: int) -> datetime:
        """Returns the start date for news collection."""
        return (datetime.now() - timedelta(days=days_ago)).replace(hour=0, minute=0, second=0, microsecond=0)

    def get_end_date(self) -> datetime:
        """Returns the end date for news collection."""
        return datetime.now().replace(hour=23, minute=59, second=59, microsecond=999999)
