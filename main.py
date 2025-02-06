import asyncio
import json
import logging
import os
from datetime import datetime
from agents.collector import CollectorAgent
from agents.manager import ManagerAgent
from agents.context_maker import ContextMakerAgent
from agents.writer import WriterAgent
from agents.publisher import PublisherAgent
from data_models import NewsDigest

# Configure basic logging
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "agents.log")

# Configure root logger
logging.basicConfig(
    level=logging.DEBUG,  # Set to DEBUG to capture all levels
    format="%(asctime)s - [%(levelname)s] - %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(log_file, mode="a", encoding="utf-8"),
        logging.StreamHandler()  # Also log to console
    ],
    force=True  # Force reconfiguration of the root logger
)

# Create main logger
logger = logging.getLogger("NewsAgents")
logger.setLevel(logging.DEBUG)

async def main():
    try:
        logger.info("🚀 Starting news processing pipeline...")
        
        # Initialize agents
        logger.debug("Initializing agents...")
        
        collector = CollectorAgent()
        logger.debug("CollectorAgent initialized")
        
        manager = ManagerAgent()
        logger.debug("ManagerAgent initialized")
        
        context_maker = ContextMakerAgent(vector_db_path="news_faiss_db")
        logger.debug("ContextMakerAgent initialized")
        
        writer = WriterAgent(
            vector_db_path="news_faiss_db",
            translate_to_russian=True
        )
        logger.debug("WriterAgent initialized")
        
        publisher = PublisherAgent()
        logger.debug("PublisherAgent initialized")
        
        logger.info("✅ All agents initialized successfully")

        # Configure news sources
        source_configs = [
            {"url": "https://feeds.bbci.co.uk/news/world/rss.xml", "days_ago": 0},
        ]

        # Collect news
        news_items = await collector.process(source_configs)
        if not news_items:
            logger.warning("⚠ No news items collected")
            return
        logger.info(f"✅ Collected {len(news_items)} news items")

        # Group news
        news_clusters = await manager.process(news_items)
        logger.info(f"✅ Grouped news into {len(news_clusters)} clusters")

        # Add context
        news_with_context = await context_maker.add_context(news_clusters)
        logger.info(f"✅ Added context to {len(news_with_context)} news items")

        # Process and analyze news
        processed_news = await writer.process(news_with_context)
        logger.info(f"✅ Processed {len(processed_news)} news items")

        # Save results
        with open("processed_news.json", "w", encoding="utf-8") as f:
            json.dump([news.model_dump() for news in processed_news], f, ensure_ascii=False, indent=2)
        logger.info("✅ Saved processed news to JSON")

        # Create and publish digest
        if processed_news:
            # Create digest with news items
            digest = NewsDigest(
                date_generated=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                region="Global",  # You can modify this based on your needs
                news_items=processed_news  # Use the processed news items directly
            )

            # Publish digest
            await publisher.process(digest)
            logger.info("✅ Published news digest to Telegram")
        
        logger.info("🎉 News processing pipeline completed successfully")

    except Exception as e:
        logger.error(f"❌ Pipeline failed: {str(e)}", exc_info=True)
        raise

# Run async process
if __name__ == "__main__":
    asyncio.run(main())
