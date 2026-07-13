# PhysioBOT — Retrieval-Augmented Physiotherapy Chatbot

PhysioBOT is a Python-based **retrieval-augmented generation (RAG)** prototype that answers physiotherapy-related questions using information retrieved from a reference PDF.

The project extracts text from a physiotherapy document, splits it into smaller chunks, converts those chunks into embeddings with Google Gemini, stores them in Pinecone, retrieves the most relevant passages for a user question, and uses Gemini to generate a clear, context-aware response.

> **Medical disclaimer:** PhysioBOT is an educational prototype, not a medical device or a substitute for a qualified physiotherapist, physician, or emergency service. Do not use its responses for diagnosis or treatment decisions.

## How it works

```text
Physiotherapy PDF
       │
       ▼
Text extraction with PyMuPDF
       │
       ▼
Chunking with RecursiveCharacterTextSplitter
       │
       ▼
Gemini text embeddings
       │
       ▼
Pinecone vector index
       │
       ▼
Similarity search for the user query
       │
       ▼
Gemini-generated answer grounded in retrieved context
```

## Features

- Extracts text from PDF documents
- Splits long text into overlapping chunks for retrieval
- Generates document and query embeddings with Google Gemini
- Stores and searches vectors using Pinecone
- Retrieves the most relevant source passages for each question
- Generates friendly, detailed answers grounded in retrieved content
- Adds a reminder to consult a qualified physiotherapist

## Technology stack

- **Python 3.12**
- **Google Gemini API** for embeddings and answer generation
- **Pinecone** for vector storage and similarity search
- **PyMuPDF** (`fitz`) for PDF text extraction
- **LangChain Text Splitters** for document chunking
- **python-dotenv** for environment-variable management
- **Jupyter Notebook** for development and experimentation

## Repository structure

```text
Physio-Chatbot/
├── NOTEBOOK/
│   └── code.ipynb          # Complete ingestion, retrieval, and generation workflow
├── Sources/
│   └── ...                 # Reference physiotherapy PDF files
├── .gitignore              # Excludes the virtual environment and .env file
└── README.md
```

## Getting started

### 1. Clone the repository

```bash
git clone https://github.com/Jayagna/Physio-Chatbot.git
cd Physio-Chatbot
```

### 2. Create and activate a virtual environment

**Windows**

```bash
python -m venv physioenv
physioenv\Scripts\activate
```

**macOS/Linux**

```bash
python3 -m venv physioenv
source physioenv/bin/activate
```

### 3. Install the dependencies

```bash
pip install jupyter streamlit google-generativeai python-dotenv pymupdf langchain-text-splitters pinecone
```

### 4. Configure environment variables

Create a `.env` file in the project root:

```env
GOOGLE_API_KEY=your_google_gemini_api_key
PINECONE_API_KEY=your_pinecone_api_key
```

The `.env` file is excluded from version control. Never commit API keys to GitHub.

### 5. Create the Pinecone index

Create a Pinecone index named:

```text
physio
```

Configure the index so that its vector dimensions are compatible with the embedding model used in the notebook. The current notebook uses:

```python
models/text-embedding-004
```

The notebook stores vectors under the namespace:

```text
ns1
```

### 6. Add the source document

Place the physiotherapy reference PDF in the `Sources` directory. Then update the PDF path in `NOTEBOOK/code.ipynb`.

Replace the existing machine-specific path:

```python
extracted_data = extract_text_from_pdf(
    r"C:\path\to\Physio-Chatbot\Sources\your-document.pdf"
)
```

A portable alternative is:

```python
from pathlib import Path

pdf_path = Path("Sources") / "your-document.pdf"
extracted_data = extract_text_from_pdf(pdf_path)
```

### 7. Run the notebook

```bash
jupyter notebook NOTEBOOK/code.ipynb
```

Run the cells in order to:

1. Load the API keys.
2. Extract text from the PDF.
3. Split the text into chunks.
4. Generate embeddings.
5. Upload the vectors to Pinecone.
6. Ask a question and generate an answer.

## RAG pipeline

### Document ingestion

The reference PDF is read using PyMuPDF. Its text is split into chunks of 300 characters with an overlap of 50 characters.

```python
RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50
)
```

### Embedding and vector storage

Each chunk is converted into a vector using Gemini's `text-embedding-004` model. The vectors and their source text are uploaded to the Pinecone `physio` index.

### Retrieval

The user's question is embedded using the same embedding model. Pinecone then returns the ten most relevant chunks from the `ns1` namespace.

### Answer generation

The retrieved passages are combined into context and passed to Gemini. The prompt instructs the model to answer in a friendly, detailed, and easy-to-understand way while reminding the user that the chatbot is not a physiotherapist.

## Example questions

```text
What exercises can help with lower-back stiffness?
```

```text
What are common causes of shoulder pain?
```

```text
How does physiotherapy support recovery after a knee injury?
```

Answers depend on the content available in the indexed reference document.

## Current limitations

- The project currently runs from a Jupyter Notebook rather than a deployed application.
- The PDF path is configured manually.
- The Pinecone index must be created before running the notebook.
- Document ingestion and question answering are part of the same notebook workflow.
- Responses may still contain incomplete or incorrect medical information.
- The project does not currently provide citations to exact PDF pages in its final answers.
- Streamlit is imported, but a Streamlit user interface has not yet been implemented.

## Potential improvements

- Build a Streamlit chat interface
- Separate document ingestion from the chat application
- Support uploading multiple PDFs
- Add source names and page numbers to retrieved chunks
- Display citations alongside generated answers
- Add conversation history
- Add input validation and API error handling
- Add automated tests and dependency pinning
- Replace machine-specific paths with project-relative paths
- Add evaluation metrics for retrieval and response quality

## Responsible use

PhysioBOT should only be used for learning and general informational purposes. Users experiencing severe pain, loss of movement, numbness, breathing difficulty, a recent serious injury, or another urgent symptom should contact an appropriate healthcare professional or emergency service.

## Author

**Jayagna Patel**

GitHub: [@Jayagna](https://github.com/Jayagna)

## License

No license has been added yet. Unless a license is provided, the repository's code remains protected by default copyright rules.
