# Legal Case Document Summarization Backend

This project is a backend server that processes and summarizes legal case documents using Natural Language Processing (NLP) models like BERT and BART. The server can be accessed via a `curl` command or Postman to query and retrieve summaries of the most relevant documents based on a given query.

## Features

- **Query Legal Documents**: Retrieve the most relevant legal case documents based on a query.
- **Document Summarization**: Generate concise summaries of the retrieved documents.
- **Preprocessing**: Text preprocessing and tokenization using BERT.
- **Similarity Matching**: Use cosine similarity to find the most relevant documents.

## Technologies Used

- **Flask**: A lightweight WSGI web application framework for Python.
- **PyTorch**: An open-source machine learning framework for NLP model inference.
- **BERT**: A transformer-based model for text preprocessing and embedding generation.
- **BART**: A transformer model for text summarization.
- **NLTK**: Natural Language Toolkit for tokenization and stemming.
- **Scikit-Learn**: Used for calculating cosine similarity between query and document embeddings.

## Setup Instructions

### Prerequisites

- Python 3.8 or higher
- Pip package manager
- `virtualenv` (optional but recommended)

### Installation

1. **Clone the repository:**

    ```bash
    git clone https://github.com/yourusername/legal-case-summarization.git
    cd legal-case-summarization
    ```

2. **Create a virtual environment (optional but recommended):**

    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3. **Install the required packages:**

    ```bash
    pip install -r requirements.txt
    ```

4. **Ensure your legal case documents are placed in the specified directory:**

    Update the `directory` variable in the script to point to your document folder:

    ```python
    directory = r"C:\path\to\your\documents"
    ```

### Running the Server

1. **Start the Flask server:**

    ```bash
    python app.py
    ```

2. The server will start running on `http://127.0.0.1:5000/`.

### Making Requests

You can interact with the server using `curl` or Postman.

#### Example `curl` Command

```bash
curl -X POST http://127.0.0.1:5000/query -H "Content-Type: application/json" -d '{"query": "Your legal query here"}'
