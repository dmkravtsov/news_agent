# News Multi-Agent System

An intelligent news processing system that collects, analyzes, and delivers news digests using a multi-agent architecture. The system processes news from various sources, adds context, performs sentiment analysis, and delivers formatted digests to Telegram.

## Features

- **Intelligent News Collection**: Collects news from multiple RSS feeds
- **Smart News Grouping**: Groups related news items using semantic analysis
- **Context Enhancement**: Adds historical context and future implications
- **Advanced Analytics**: Provides sentiment analysis and key insights
- **Multi-language Support**: Translates content to Russian
- **Telegram Integration**: Delivers formatted digests to Telegram channels

## System Architecture

The system uses a multi-agent architecture with specialized agents:

1. **Collector Agent**: Fetches news from configured RSS feeds
2. **Manager Agent**: Groups related news items using semantic clustering
3. **Context Maker Agent**: Enhances news with historical context using vector database
4. **Writer Agent**: Processes and analyzes news, including translation and sentiment analysis
5. **Publisher Agent**: Formats and publishes digests to Telegram

## Installation

1. Clone the repository:
```bash
git clone [your-repo-url]
cd news_multi_agent
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Unix/MacOS
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables in `.env` file:
```env
TELEGRAM_BOT_TOKEN=your_telegram_bot_token
TELEGRAM_CHAT_ID=your_telegram_chat_id
OPENAI_API_KEY=your_openai_api_key
```

## Usage

1. Configure news sources in `main.py`:
```python
source_configs = [
    {"url": "https://feeds.bbci.co.uk/news/world/rss.xml", "days_ago": 0},
    # Add more sources as needed
]
```

2. Run the application:
```bash
python main.py
```

## Sample Output

The system generates structured news digests in the following format:

```
 News Digest
 Generated: 2025-02-06 11:50:29
 Region: Global

*1. [News Description]*
 Category: Politics
 Tags: diplomacy, international relations, summit

 Analysis:
[Analysis of the news and its significance]

 Historical Context:
[Events and context leading to this news]

 Future Implications:
[Predicted future developments]
 Sentiment Score: 0.75

────────────────────────

*2. [Next News Description]*
...
```

## Technologies Used

### Core Technologies
- Python 3.11+
- Pydantic for data validation
- AsyncIO for asynchronous operations
- FAISS for vector similarity search
- Telegram Bot API for message delivery

### AI and NLP
- OpenAI API for text analysis
- Sentence Transformers for semantic analysis
- NLTK for text processing
- MCP Server for efficient RAG (Retrieval-Augmented Generation)
  - Optimized vector storage and retrieval
  - Fast similarity search for context enhancement
  - Efficient document chunking and embedding

### Data Storage
- FAISS vector database for context storage
- JSON for processed news storage

### Development Tools
- Poetry for dependency management
- Python-dotenv for environment management
- Logging for system monitoring

## Project Structure

```
news_multi_agent/
├── agents/
│   ├── collector.py
│   ├── manager.py
│   ├── context_maker.py
│   ├── writer.py
│   └── publisher.py
├── base_agent.py
├── data_models.py
├── main.py
├── requirements.txt
└── .env
```

## Logging

The system maintains detailed logs in the `logs/agents.log` file, capturing:
- Agent initialization and operations
- News processing steps
- Errors and warnings
- Performance metrics

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

MIT License

Copyright (c) 2025 Dmitriy Kravtsov

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## Author

Dmitriy Kravtsov dmkravtsov@gmail.com
