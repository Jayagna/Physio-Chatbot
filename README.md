# 🧠 PhysioBOT

### A Retrieval-Augmented Physiotherapy Chatbot

<p>
  <strong>Ask physiotherapy-related questions and receive context-aware answers grounded in a reference document.</strong>
</p>

<p>
  <img src="https://img.shields.io/badge/Python-3.12-blue?logo=python&logoColor=white" alt="Python 3.12">
  <img src="https://img.shields.io/badge/Google-Gemini-8E75B2?logo=google&logoColor=white" alt="Google Gemini">
  <img src="https://img.shields.io/badge/Pinecone-Vector%20Database-000000?logo=pinecone&logoColor=white" alt="Pinecone">
  <img src="https://img.shields.io/badge/LangChain-Text%20Splitting-1C3C3C" alt="LangChain">
  <img src="https://img.shields.io/badge/PyMuPDF-PDF%20Processing-red" alt="PyMuPDF">
  <img src="https://img.shields.io/badge/Status-Prototype-orange" alt="Prototype">
</p>

</div>

---

## ✨ Overview

**PhysioBOT** is a Python-based **Retrieval-Augmented Generation (RAG)** prototype designed to answer physiotherapy-related questions using information retrieved from a reference PDF.

Instead of relying only on a language model's general knowledge, PhysioBOT:

1. Extracts text from a physiotherapy document  
2. Splits the text into smaller chunks  
3. Converts each chunk into an embedding  
4. Stores the embeddings in Pinecone  
5. Retrieves the most relevant passages for a user query  
6. Uses Google Gemini to generate a detailed response grounded in the retrieved content  

> [!CAUTION]
> **Medical disclaimer:** PhysioBOT is an educational prototype. It is not a medical device and should not be used as a replacement for a qualified physiotherapist, physician, or emergency service.

---

## 🚀 Key Features

<table>
<tr>
<td width="50%">

### 📄 Document Processing
- Extracts text from physiotherapy PDFs
- Handles long documents through chunking
- Uses overlapping chunks for better context retention

</td>
<td width="50%">

### 🔎 Semantic Retrieval
- Generates embeddings using Google Gemini
- Stores vectors in Pinecone
- Retrieves the most relevant passages for each query

</td>
</tr>

<tr>
<td width="50%">

### 💬 Context-Aware Answers
- Generates detailed, easy-to-understand responses
- Grounds answers in retrieved source material
- Includes a reminder to consult a professional

</td>
<td width="50%">

### 🧪 Experiment-Friendly
- Built inside a Jupyter Notebook
- Easy to inspect and modify
- Suitable for learning RAG concepts

</td>
</tr>
</table>

---

## 🏗️ Architecture

```mermaid
flowchart TD
    A[Physiotherapy PDF] --> B[Text Extraction<br/>PyMuPDF]
    B --> C[Text Chunking<br/>LangChain]
    C --> D[Embeddings<br/>Gemini text-embedding-004]
    D --> E[(Pinecone Vector Database)]
    F[User Question] --> G[Query Embedding]
    G --> E
    E --> H[Relevant Chunks]
    H --> I[Gemini LLM]
    F --> I
    I --> J[Context-Aware Answer]
