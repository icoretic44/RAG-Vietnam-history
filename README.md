
---

# RAG Pipeline for Vietnam's Resistance History

This repository implements a **Retrieval-Augmented Generation (RAG)** pipeline to explore and analyze Vietnam's resistance battles against the French Empire. The RAG system integrates **Gemini API** calls and **Tavily Search** to embed and retrieve information from both local documents and the internet.

The pipeline allows users to extract data from PDF files, embed it into a vector database (Chroma), and perform searches or retrievals to generate meaningful insights using advanced retrieval techniques.

## Overview

The RAG pipeline is designed to:
- Extract and embed historical data from PDFs into a **Chroma** vector database.
- Retrieve relevant historical content based on user queries.
- Search the internet via **Gemini API** and **Tavily Search** for supplemental information.
- Provide a simple interface, powered by **Streamlit**, for users to interact with the data and explore Vietnam's history in detail.

## Features
- **Data Extraction**: Parse historical data from provided PDF files.
- **Embedding**: Embed extracted text into a vector database for efficient retrieval.
- **Search and Retrieval**: Use advanced search mechanisms (Gemini API, Tavily Search) to find relevant information locally and online.
- **User Interface**: A Streamlit-based interface for user-friendly interaction with the RAG pipeline.

## Getting Started

Follow the steps below to set up and run the pipeline.

### Prerequisites
- Install Python (version 3.7 or higher).
- Ensure you have API keys for **Gemini** and **Tavily Search**.

---

### Guide to Set Up

1. **Create a Virtual Environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # For Linux/MacOS
   venv\Scripts\activate     # For Windows
   ```

2. **Configure Environment Variables**:
   - Create a `.env` file in the root directory.
   - Add the necessary API keys and configurations. For example:
     ```
     GEMINI_API_KEY=your_gemini_api_key
     TAVILY_API_KEY=your_tavily_api_key
     ```

3. **Install Dependencies**:
   - Install all required Python packages using `requirements.txt`:
     ```bash
     pip install -r requirements.txt
     ```

4. **Extract and Embed Data**:
   - Use the `extract_data.py` script to process PDF files and embed them into the Chroma vector database:
     ```bash
     python extract_data.py
     ```

5. **Run the RAG Pipeline**:
   - Start the Streamlit app to interact with the RAG pipeline:
     ```bash
     streamlit run app.py
     ```

---

## Usage

Once the pipeline is running, you can:
- Upload PDFs to extract and embed historical data.
- Input your queries to retrieve relevant information from the vector database.
- Supplement your search with internet-based retrieval using **Gemini API** and **Tavily Search**.

The user interface provides a seamless way to interact with the data.

---

## Acknowledgments

This repository is an effort to showcase Vietnam's rich history through advanced retrieval and generation techniques. Special thanks to the creators of **Gemini API**, **Tavily Search**, and **Chroma** for their powerful tools that make this project possible.

---
