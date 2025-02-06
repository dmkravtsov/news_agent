from base_agent import BaseAgent
from data_models import NewsDigest
from pydantic import Field, PrivateAttr
import os
from telegram import Bot
from typing import Optional
import asyncio

class PublisherAgent(BaseAgent):
    """Agent for publishing news digests to Telegram channel."""
    
    name: str = "PublisherAgent"  # Set name as class field
    telegram_bot_token: str = Field(default_factory=lambda: os.getenv("TELEGRAM_BOT_TOKEN"))
    chat_id: str = Field(default_factory=lambda: os.getenv("TELEGRAM_CHAT_ID"))
    _bot: Optional[Bot] = PrivateAttr(default=None)

    def __init__(self, **data):
        """Initialize agent using Pydantic validation."""
        super().__init__(**data)  # Ensures Pydantic handles field validation

        # Check required environment variables
        if not self.telegram_bot_token or not self.chat_id:
            self.log("❌ Telegram bot token or chat ID not found in environment variables", level="ERROR")
            raise ValueError("Telegram bot token or chat ID not found in environment variables.")

        # Initialize bot
        try:
            object.__setattr__(self, "_bot", Bot(token=self.telegram_bot_token))
            self.log("✅ Telegram bot initialized")
        except Exception as e:
            self.log(f"❌ Failed to initialize Telegram bot: {str(e)}", level="ERROR")
            raise

    async def process(self, data: NewsDigest) -> None:
        """
        Publishes digest to Telegram channel.

        Args:
            data: NewsDigest object to publish.
        """
        self.log(f"""🔄 Preparing to publish digest:
            - Date: {data.date_generated}
            - Region: {data.region if data.region else 'Global'}
            - News Items: {len(data.news_items)} items""")
        
        try:
            message = self._format_message(data)
            message_length = len(message)
            self.log(f"✅ Message formatted successfully (Length: {message_length} characters)")
            
            await self._send_message(message)
            self.log("✅ Digest published to Telegram")
        except Exception as e:
            self.log(f"❌ Failed to publish digest: {str(e)}", level="ERROR")
            self.log(f"Error details: {type(e).__name__}", level="ERROR")
            raise

    def _format_message(self, digest: NewsDigest) -> str:
        """
        Formats digest for Telegram publication by grouping each NewsItem's data.

        Args:
            digest: NewsDigest object to format.

        Returns:
            Formatted message string.
        """
        try:
            self.log("🔄 Starting message formatting...", level="INFO")
            message_parts = []
            
            # Add header
            message_parts.append(f"📰 *News Digest*")
            message_parts.append(f"🕒 *Generated:* {digest.date_generated}")
            if digest.region:
                message_parts.append(f"📍 *Region:* {digest.region}")

            # Format each news item
            for idx, item in enumerate(digest.news_items, 1):
                self.log(f"Processing news item {idx}", level="DEBUG")
                
                # Add separator between items
                if idx > 1:
                    message_parts.append("\n" + "─" * 30 + "\n")

                # Main news block
                message_parts.extend([
                    f"*{idx}. {item.description}*",
                    f"📊 *Category:* {item.category}",
                    f"🏷️ *Tags:* {', '.join(item.tags[:3])}"  # Limit to 3 tags
                ])

                # Analysis block
                message_parts.extend([
                    "",  # Empty line for spacing
                    f"📈 *Analysis:*",
                    item.analytics,
                    "",  # Empty line for spacing
                    f"📜 *Historical Context:*",
                    item.precursors,
                    "",  # Empty line for spacing
                    f"🔮 *Future Implications:*",
                    item.future_trend
                ])

                # Sentiment
                sentiment_emoji = "😊" if item.sentiment_score > 0.6 else "😐" if item.sentiment_score > 0.4 else "😔"
                message_parts.append(f"{sentiment_emoji} *Sentiment Score:* {item.sentiment_score:.2f}")
                
                self.log(f"Completed formatting news item {idx}", level="DEBUG")
            
            formatted_message = "\n".join(message_parts)
            self.log(f"✅ Message formatting completed (Length: {len(formatted_message)} characters)", level="INFO")
            return formatted_message
            
        except Exception as e:
            self.log(f"❌ Failed to format message: {str(e)}", level="ERROR")
            self.log(f"Error details: {type(e).__name__} at {e.__traceback__.tb_lineno}", level="ERROR")
            raise

    async def _send_message(self, message: str) -> None:
        """
        Sends message to Telegram channel. If message is too long, splits it into multiple parts.

        Args:
            message: Message text to send.
        """
        try:
            # Split message into parts if it's too long (Telegram limit is 4096 chars)
            MAX_LENGTH = 4000  # Leave some room for formatting
            message_parts = []
            
            # Split message by sections (split on double newline)
            sections = message.split("\n\n")
            current_part = []
            current_length = 0
            
            for section in sections:
                # Add section length plus newlines
                section_length = len(section) + 2
                
                # If adding this section would exceed limit, save current part and start new one
                if current_length + section_length > MAX_LENGTH:
                    message_parts.append("\n\n".join(current_part))
                    current_part = []
                    current_length = 0
                
                current_part.append(section)
                current_length += section_length
            
            # Add any remaining content as the last part
            if current_part:
                message_parts.append("\n\n".join(current_part))
            
            # Log message splitting info
            total_parts = len(message_parts)
            if total_parts > 1:
                self.log(f"📨 Message split into {total_parts} parts:")
                for i, part in enumerate(message_parts, 1):
                    self.log(f"  - Part {i}: {len(part)} characters")
            
            # Send each part with pagination
            for i, part in enumerate(message_parts, 1):
                # Add pagination footer if multiple parts
                if total_parts > 1:
                    part += f"\n\n📄 Part {i}/{total_parts}"
                
                self.log(f"🔄 Sending part {i}/{total_parts}...")
                await self._bot.send_message(
                    chat_id=self.chat_id,
                    text=part,
                    parse_mode="Markdown"
                )
                self.log(f"✅ Part {i}/{total_parts} sent successfully")
                
                # Add small delay between messages to maintain order
                if i < total_parts:
                    await asyncio.sleep(1)
                
            self.log(f"✅ Complete message sent successfully in {total_parts} part(s)")
        except Exception as e:
            self.log(f"❌ Failed to send message: {str(e)}", level="ERROR")
            self.log(f"Error details: {type(e).__name__}", level="ERROR")
            raise
