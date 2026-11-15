# Positive Parenting RAG Assistant

A retrieval-augmented generation (RAG) chatbot that answers questions using the
PDF documents in `positive_parenting/`. It uses LangChain to download, load, and
split the PDFs, OpenAI to create embeddings and generate answers, Pinecone for
vector search, and Gradio for the chat interface.

## How it works

When `app.py` starts, it:

1. Creates the Pinecone index `langchain-index` if it does not already exist.
   The index is configured for 1536-dimensional vectors, cosine similarity, and
   AWS `us-east-1`.
2. Downloads each configured PDF if its corresponding file is not already in
   `positive_parenting/`, then loads every PDF in that folder and splits the
   text into 700-character chunks with 50-character overlap.
3. Embeds the chunks using OpenAI's `text-embedding-3-small` model and adds
   them to Pinecone.
4. Starts a Gradio chat interface. For each question, it retrieves the six
   most similar chunks and asks the `gpt-6.1-sol` chat model to answer clear and concise answer. If the context does not contain an answer, the prompt
   tells the model to say it cannot find one.

The chat UI saves conversation history, but the current retrieval chain answers
each message using that message alone; it does not use earlier turns as context.
The chat response includes the retrieved source filenames and page numbers.
Repeated references to the same source page are listed only once.

### PDF sources

The app downloads these PDFs into `positive_parenting/` when they are not
already present:

- [Pillars for Success — Positive Parenting](https://www.pillarsforsuccess.com/wp-content/uploads/2021/03/Pillars-for-Success-Positive-Parenting.pdf)
- [NICRO — Positive Parenting Workbook](https://www.nicro.org.za/images/PDFs/Positive%20Parenting%20Workbook.pdf)
- [Positive Parenting PDF](https://cdn.bookey.app/files/pdf/book/en/positive-parenting.pdf)

The app processes every PDF in `positive_parenting/`, including any additional
PDFs you place there.

## Requirements

- Python
- An OpenAI API key
- A Pinecone API key and a Pinecone account that can create a serverless index

## Setup

Run these commands from the project directory in PowerShell:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Create a `.env` file in the project root and set your credentials. The app reads
these values with `python-dotenv`:

```dotenv
OPENAI_API_KEY=your_openai_api_key
PINECONE_API_KEY=your_pinecone_api_key
```

Keep the real `.env` file private; do not commit API keys to source control.

## Run

Run the app from the project directory:

```powershell
python app.py
```

Open the local URL printed by Gradio in your browser and ask a question about
the documents.

## Important notes

- The app checks whether each downloaded PDF already exists, so it does not
  download those files again unless they are removed.
- The app upserts PDF chunks to Pinecone using stable IDs based on each chunk's
  source, page, and text, so restarting does not create duplicate vectors.
  Existing chunks are still re-embedded and upserted, which incurs embedding/API
  usage. Vectors created by earlier versions with generated IDs are not removed
  automatically.
- The Pinecone index name, region, embedding dimensions, chunk size, retrieval
  count, and model names are configured directly in `app.py`.
- The app requires working OpenAI and Pinecone credentials and network access.
  Network access is also needed to download the PDFs when they are not present.

## Project layout

```text
.
├── app.py
├── Requirements.txt
└── positive_parenting/
    ├── Pillars-for-Success-Positive-Parenting.pdf
    ├── Positive%20Parenting%20Workbook.pdf
    └── positive-parenting.pdf
```
