import hashlib
import html
import json
import os
import urllib.request
from pathlib import Path

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_classic.chains import RetrievalQA
from langchain_classic.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec


FILES_LIST = [
    "https://www.pillarsforsuccess.com/wp-content/uploads/2021/03/Pillars-for-Success-Positive-Parenting.pdf",
    "https://www.nicro.org.za/images/PDFs/Positive%20Parenting%20Workbook.pdf",
    "https://cdn.bookey.app/files/pdf/book/en/positive-parenting.pdf",
]
FOLDER_PATH = "./positive_parenting/"
DOWNLOAD_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/91.0.4472.124 Safari/537.36"
    )
}
INDEX_NAME = "langchain-index"
GREETINGS = {
    "hi",
    "hello",
    "hey",
    "hi there",
    "hello there",
    "hey there",
    "good morning",
    "good afternoon",
    "good evening",
}
GREETING_RESPONSE = "Hello! Ask me a question about the parenting documents."
PROMPT_TEMPLATE = """
You are a document-based question answering assistant.

Rules:
1. Answer using ONLY the provided context.
2. Do not invent information or use outside knowledge.
3. If the answer is not supported by the context,
   say: "I couldn't find this information in the documents."
4. If documents contradict each other, explain the conflict.
5. Keep the answer clear and concise.

Context:
{context}

Question:
{question}

Answer:
"""


def download_files(files_list, folder_path, headers):
    folder_path = Path(folder_path)
    folder_path.mkdir(parents=True, exist_ok=True)
    for url in files_list:
        file_path = folder_path / url.rpartition("/")[2]
        if file_path.exists():
            print(f"File already exists: {file_path}")
            continue

        request = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(request) as response, file_path.open("wb") as out_file:
            out_file.write(response.read())
        print(f"Downloaded: {file_path}")


def split_documents(folder_path):
    documents = PyPDFDirectoryLoader(folder_path).load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=50)
    return splitter.split_documents(documents)


def document_id(document):
    identity = json.dumps(
        [
            document.metadata.get("source"),
            document.metadata.get("page"),
            document.page_content,
        ],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


def create_rag_chain():
    load_dotenv()
    pinecone_api_key = os.environ["PINECONE_API_KEY"]
    pinecone = Pinecone(pinecone_api_key)

    if not pinecone.has_index(INDEX_NAME):
        pinecone.create_index(
            name=INDEX_NAME,
            dimension=1536,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )

    download_files(FILES_LIST, FOLDER_PATH, DOWNLOAD_HEADERS)
    documents = split_documents(FOLDER_PATH)

    vectorstore = PineconeVectorStore(
        index=pinecone.Index(INDEX_NAME),
        embedding=OpenAIEmbeddings(model="text-embedding-3-small"),
    )
    document_ids = [document_id(document) for document in documents]
    vectorstore.add_documents(documents, ids=document_ids)
    print(f"Upserted {len(documents)} chunks with stable IDs.")

    prompt = PromptTemplate(
        template=PROMPT_TEMPLATE,
        input_variables=["context", "question"],
    )
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 6},
    )
    return RetrievalQA.from_chain_type(
        llm=init_chat_model("gpt-6.1-sol"),
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt},
    )


def source_title(source_path):
    source_name = os.path.basename(source_path) if source_path else "Unknown source"
    friendly_names = {
        "Pillars-for-Success-Positive-Parenting.pdf": "Pillars for Success",
        "Positive%20Parenting%20Workbook.pdf": "Positive Parenting Workbook",
        "positive-parenting.pdf": "Positive Parenting",
    }
    if source_name in friendly_names:
        return friendly_names[source_name]

    cleaned_name = source_name.replace("%20", " ").replace("-", " ")
    return cleaned_name.replace(".pdf", "").strip().title()


def format_sources(sources):
    if not sources:
        return ""

    source_pills = [
        f'<span class="source-pill">{html.escape(source)}</span>'
        for source in sources
    ]
    return (
        '<div class="source-box">'
        '<div class="source-label">Sources</div>'
        f'<div class="source-list">{"".join(source_pills)}</div>'
        "</div>"
    )


def is_greeting(message):
    normalized_message = message.casefold().strip().rstrip("!.,?").strip()
    return normalized_message in GREETINGS


def create_chat_handler(retrieval_qa):
    def chat_with_model(new_message, _=None):
        if not new_message.strip():
            return "Please enter a message."
        if is_greeting(new_message):
            return GREETING_RESPONSE

        result = retrieval_qa.invoke({"query": new_message})
        answer = result["result"].strip()
        sources = []
        seen_sources = set()

        for document in result["source_documents"]:
            metadata = document.metadata
            source_name = source_title(metadata.get("source"))
            page_number = metadata.get("page_label")
            page = metadata.get("page")
            if page_number is None and isinstance(page, int):
                page_number = page + 1

            citation = (
                f"{source_name} · Page {page_number}"
                if page_number is not None
                else source_name
            )
            if citation not in seen_sources:
                sources.append(citation)
                seen_sources.add(citation)

        answer_has_no_match = (
            "i couldn't find this information in the documents" in answer.casefold()
        )
        if sources and not answer_has_no_match:
            answer += format_sources(sources)
        return answer

    return chat_with_model
