# AI Academy RAG Chatbot

A Python 3.12-based chatbot application using Retrieval-Augmented Generation (RAG) to provide intelligent responses based on document knowledge bases.

## Features

- **Document Processing**: Load and process PDF documents
- **Vector Embeddings**: Generate and store embeddings for efficient retrieval
- **RAG Pipeline**: Retrieve relevant context and generate responses using OpenAI
- **Voice Integration**: Whisper support for audio transcription

## Project Structure

```
ai-academy-rag-chatbot/
├── src/                    # Source code modules
│   ├── __init__.py
│   ├── chatbot.py         # Main chatbot logic
│   ├── embeddings.py      # Vector embeddings handling
│   ├── data_loader.py     # Document loading and processing
│   └── retriever.py       # Document retrieval logic
├── data/                   # Store your documents here
├── embeddings/            # Vector database storage
├── .env.example           # Environment variables template
├── .gitignore            # Git ignore rules
├── requirements.txt      # Python dependencies
└── README.md             # This file
```

## Prerequisites

- Python 3.12 or higher
- OpenAI API key (get one from [OpenAI Platform](https://platform.openai.com/))

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/Votok/ai-academy-rag-chatbot.git
cd ai-academy-rag-chatbot
```

### 2. Create a Virtual Environment

Create and activate a virtual environment to isolate project dependencies:

**On Linux/MacOS:**

```bash
python3 -m venv venv
source venv/bin/activate
```

**On Windows:**

```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies

Install all required packages from requirements.txt:

```bash
pip install -r requirements.txt
```

This will install:

- `langchain` - Framework for building LLM applications
- `openai` - OpenAI API client
- `chromadb` - Vector database for embeddings
- `openai-whisper` - Speech recognition
- `pypdf` - PDF document processing
- `tiktoken` - Token counting for OpenAI models
- `python-dotenv` - Environment variable management

### 4. Configure Environment Variables

Create a `.env` file from the template:

```bash
cp .env.example .env
```

Edit the `.env` file and add your OpenAI API key:

```
OPENAI_API_KEY=sk-your-actual-api-key-here
```

⚠️ **Important**: Never commit your `.env` file to version control. It's already listed in `.gitignore`.

## Usage

(Coming soon - implementation in progress)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is open source and available under the MIT License.
