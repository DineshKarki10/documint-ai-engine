# DocuMint: Production-Grade RAG & Evaluation Engine

DocuMint is an enterprise document intelligence architecture built for Retrieval-Augmented Generation (RAG) workflows. It enables users to upload technical PDFs, ingest them into a local vector database (ChromaDB), query content through a dual-provider LLM pipeline, and run automated evaluation checks to measure hallucination resistance, answer relevance, and contextual recall.

## Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Installation](#installation)
- [Usage](#usage)
- [API Reference](#api-reference)
- [Environment Variables](#environment-variables)
- [Evaluation Metrics](#evaluation-metrics)
- [Project Structure](#project-structure)
- [Hugging Face Spaces](#hugging-face-spaces)
- [Contributing](#contributing)
- [License](#license)

## Features

- **PDF Document Ingestion**: Upload and parse technical PDF documentation into semantic chunks using LangChain text splitters
- **Local Vector Database**: ChromaDB persistence with Hugging Face `all-MiniLM-L6-v2` embeddings for local semantic search
- **Dual-Provider LLM Fallback**: Primary Groq (Llama 3.3 70B) with automatic fallback to Hugging Face Serverless (Llama 3.1 8B) on rate limits or failures
- **Gradio Web Interface**: Multi-tab UI for intelligent Q&A chat, document ingestion, and automated evaluation reporting
- **Automated RAG Evaluation**: Programmatic metrics for faithfulness, answer relevance, and context recall
- **Source Attribution**: Query responses include retrieved document sources for transparency

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      Gradio Web Interface                        │
│  [Q&A Chat] [Document Ingestion] [Evaluation Dashboard]         │
└─────────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────────┐
│                    RAG Pipeline (retriever.py)                   │
│  1. Similarity Search → ChromaDB                                 │
│  2. Context Assembly → LLM Generation                            │
└─────────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────────┐
│              Dual-Provider Inference Fallback                    │
│  Groq (llama-3.3-70b-versatile) → HF Serverless (Llama 3.1 8B) │
└─────────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────────┐
│                    Vector Store (ChromaDB)                       │
│  Embeddings: sentence-transformers/all-MiniLM-L6-v2             │
└─────────────────────────────────────────────────────────────────┘
```

### Data Flow

1. **Ingestion**: User uploads a PDF → `PyPDFLoader` extracts pages → `RecursiveCharacterTextSplitter` chunks text (1000 chars, 200 overlap) → chunks stored in ChromaDB with source metadata
2. **Retrieval**: User query → `similarity_search` finds top-k chunks → context assembled → sources tracked
3. **Generation**: Context + query → Groq primary API → fallback to Hugging Face on failure → response with source attribution
4. **Evaluation**: Simulated metrics for faithfulness, relevance, and recall across 5 test cases

## Tech Stack

| Layer | Technology | Version |
|-------|-----------|---------|
| UI Framework | Gradio | >= 5.0.0 |
| LLM Framework | LangChain | >= 0.2.0 |
| Vector Database | ChromaDB | >= 0.5.0 |
| PDF Parsing | PyPDF | >= 4.0.0 |
| Embeddings | sentence-transformers | >= 3.0.0 |
| LLM Provider (Primary) | Groq | >= 0.5.0 |
| LLM Provider (Fallback) | Hugging Face Hub | >= 0.23.0 |
| Validation | Pydantic | >= 2.0.0 |
| Runtime | PyTorch (CPU) | >= 2.6.0 |
| Config | python-dotenv | >= 1.0.0 |

## Installation

### Prerequisites

- Python 3.10 or higher
- Groq API key (primary inference) — [Get one here](https://console.groq.com)
- Hugging Face token (fallback inference) — [Get one here](https://huggingface.co/settings/tokens)

### Quick Start

```bash
# Clone the repository
git clone <repository-url>
cd documint-ai-engine

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate      # Linux/macOS
.venv\Scripts\activate         # Windows

# Install dependencies
pip install -r requirements.txt

# Set environment variables
export GROQ_API_KEY="your_groq_api_key"
export HF_TOKEN="your_huggingface_token"   # Optional fallback

# Run the application
python run.py
```

The Gradio interface will be available at `http://0.0.0.0:7860` (or the port specified via `--port` or the `PORT` environment variable).

### Docker

```bash
# Build the image
docker build -t documint-ai-engine .

# Run the container
docker run -p 7860:7860 \
  -e GROQ_API_KEY=your_groq_api_key \
  -e HF_TOKEN=your_huggingface_token \
  documint-ai-engine
```

## Hugging Face Spaces

DocuMint can be deployed directly to [Hugging Face Spaces](https://huggingface.co/spaces) using the `space.yaml` configuration file at the repository root.

### `space.yaml`

```yaml
sdk: docker
app_port: 7860
title: DocuMint Engine
emoji: 📄
colorFrom: purple
colorTo: indigo
pinned: false
license: mit
```

### Configuration Reference

| Field | Value | Description |
|-------|-------|-------------|
| `sdk` | `docker` | Uses Docker as the runtime SDK |
| `app_port` | `7860` | Gradio server port exposed by the container |
| `title` | `DocuMint Engine` | Display name for the Space |
| `emoji` | `📄` | Emoji shown next to the Space title |
| `colorFrom` | `purple` | Gradient start color for the Space header |
| `colorTo` | `indigo` | Gradient end color for the Space header |
| `pinned` | `false` | Whether the Space is pinned on your profile |
| `license` | `mit` | License identifier for the Space |

### Deploying to Hugging Face Spaces

1. Create a new Space on [Hugging Face](https://huggingface.co/new-space)
2. Select **Docker** as the SDK
3. Connect your repository or upload the files directly
4. Add the following secrets in the Space settings:
   - `GROQ_API_KEY` — Your Groq API key (required)
   - `HF_TOKEN` — Your Hugging Face token (optional fallback)
5. The Space will build the Docker image and launch on port 7860

## Usage

### Document Ingestion

1. Launch the app: `python run.py`
2. Navigate to the **Document Ingestion** tab
3. Upload a PDF technical document (manuals, reports, documentation)
4. Click **Ingest & Embed Chunks** to parse and store the document in ChromaDB

### Intelligent Q&A

1. Navigate to the **Intelligent Q&A Chat** tab
2. Ask questions about your ingested documentation
3. Responses include source document attributions and the inference provider used

### Automated Evaluation

1. Navigate to the **Evaluation Dashboard** tab
2. Click **Run Automated Evaluation Suite**
3. Review metrics: Faithfulness, Answer Relevance, and Context Recall scores

### CLI Options

```bash
python run.py               # Default port 7860
python run.py --port 8080   # Custom port
```

## API Reference

### `run.py`

#### `main()`
Loads environment variables from `.env`, configures the port, and launches the Gradio app via `app.demo`.

**CLI Arguments:**
- `--port PORT`: Override the default port (7860)

### `app.py`

#### `handle_file_upload(file)`
Handles PDF file upload, chunking, and persistence into ChromaDB.

**Parameters:**
- `file`: Uploaded PDF file object from Gradio

**Returns:** Status message with chunk count

#### `handle_query(message, history)`
Executes RAG retrieval and dual-provider generation for the chat interface.

**Parameters:**
- `message`: User query string
- `history`: Chat history list

**Returns:** Formatted response with source attributions

#### `handle_evaluation()`
Triggers the automated RAG evaluation suite and renders metrics.

**Returns:** Markdown-formatted evaluation report

### `src/ingestion.py`

#### `process_pdf(file_path: str, db: Chroma) -> int`
Loads a PDF file, splits it into semantic chunks, and adds them to the Chroma vector store.

**Parameters:**
- `file_path`: Path to the PDF file
- `db`: ChromaDB vector store instance

**Returns:** Number of chunks added

### `src/retriever.py`

#### `initialize_vector_db()`
Initializes local ChromaDB using Hugging Face sentence-transformers embeddings.

**Returns:** ChromaDB instance

#### `generate_llm_response(prompt: str, context: str) -> tuple[str, str]`
Attempts generation via Groq primary API with Hugging Face Serverless fallback.

**Parameters:**
- `prompt`: User query
- `context`: Retrieved document context

**Returns:** Tuple of (response text, provider used)

#### `query_rag_pipeline(query_text: str, db: Chroma)`
Executes vector similarity search and passes context to the LLM runner.

**Parameters:**
- `query_text`: User query string
- `db`: ChromaDB vector store instance

**Returns:** Tuple of (formatted answer, list of source files)

### `src/evaluation.py`

#### `run_rag_evaluation(db)`
Simulates automated evaluation metrics to check for system hallucinations.

**Parameters:**
- `db`: ChromaDB vector store instance

**Returns:** Dictionary with test cases evaluated, faithfulness, relevance, and recall scores

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `GROQ_API_KEY` | Yes | Groq API key for primary LLM inference |
| `HF_TOKEN` | No | Hugging Face token for fallback inference |
| `PORT` | No | Server port (default: 7860) |

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key
HF_TOKEN=your_huggingface_token
PORT=7860
```

## Evaluation Metrics

| Metric | Range | Description |
|--------|-------|-------------|
| Faithfulness | 0.0 – 1.0 | Measures hallucination alignment with retrieved context |
| Answer Relevance | 0.0 – 1.0 | Measures precision of the response against the user query |
| Context Recall | 0.0 – 1.0 | Measures completeness of retrieved document chunks |

## Project Structure

```
documint-ai-engine/
├── app.py                      # Gradio application entry point
├── Dockerfile                  # Container configuration
├── requirements.txt            # Python dependencies
├── run.py                      # Application launcher with .env loading
├── space.yaml                  # Hugging Face Spaces configuration
├── .gitignore                  # Git ignore rules
├── .dockerignore               # Docker ignore rules
├── .env                        # Local environment secrets (not committed)
├── README.md                   # Project documentation
├── src/
│   ├── __init__.py             # Package initializer
│   ├── ingestion.py            # PDF loading and chunking logic
│   ├── retriever.py            # RAG retrieval and LLM inference
│   └── evaluation.py           # RAG evaluation metrics
└── chroma_db/                  # Local vector database storage (runtime)
```

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature-name`)
3. Commit your changes (`git commit -m 'Add some feature'`)
4. Push to the branch (`git push origin feature/your-feature-name`)
5. Open a Pull Request

Please ensure your code follows the existing style and includes appropriate tests.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [Gradio](https://www.gradio.app/) — Web interface framework
- [LangChain](https://python.langchain.com/) — LLM application framework
- [ChromaDB](https://www.trychroma.com/) — Vector database
- [Groq](https://groq.com/) — Fast LLM inference
- [Hugging Face](https://huggingface.co/) — Open source AI models and inference API