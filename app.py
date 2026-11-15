import os
import html
import hashlib
import json
from pinecone import Pinecone, ServerlessSpec
import gradio as gr
import urllib.request
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.chains import RetrievalQA
from langchain_classic.prompts import PromptTemplate
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv

load_dotenv()

pinecone_api_key = os.environ["PINECONE_API_KEY"]
openAI_api_key = os.environ["OPENAI_API_KEY"]
files_list = [
        "https://www.pillarsforsuccess.com/wp-content/uploads/2021/03/Pillars-for-Success-Positive-Parenting.pdf",
        "https://www.nicro.org.za/images/PDFs/Positive%20Parenting%20Workbook.pdf",
        "https://cdn.bookey.app/files/pdf/book/en/positive-parenting.pdf",
]
folder_name = "positive_parenting"
headers_dic = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"}
folder_path ="./positive_parenting/"

def download_files(files_list, folder_name, headers_dic):
    os.makedirs(folder_name, exist_ok=True)
    for url in files_list:
        file_path = os.path.join(folder_name, url.rpartition("/")[2])
        #prevents downloading the files everytime we run the app
        if os.path.exists(file_path):
            print(f"File already exists: {file_path}")
            continue
        req = urllib.request.Request(url, headers=headers_dic)
        with urllib.request.urlopen(req) as response, open(file_path, 'wb') as out_file:
            out_file.write(response.read())
            print(f"Downloaded: {file_path}")

def split_documents(folder_path):
    loader = PyPDFDirectoryLoader(folder_path)
    docs_before_split = loader.load()
    text_splitter =  RecursiveCharacterTextSplitter(chunk_size =700,chunk_overlap = 50 )
    docs_after_split = text_splitter.split_documents(docs_before_split)
    print(f'after split: {len(docs_after_split[0].page_content)} ')
    avg_doc_length = lambda docs: sum([len(doc.page_content) for doc in docs])//len(docs)
    avg_char_before_split = avg_doc_length(docs_before_split)
    avg_char_after_split = avg_doc_length(docs_after_split)
    print(f'avg before split: {avg_char_before_split}')
    print(f'avg after split: {avg_char_after_split}')
    return docs_after_split


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


#create pinecone index
pc = Pinecone(pinecone_api_key)
index_name = "langchain-index"

if not pc.has_index(index_name):
    pc.create_index(
        name=index_name,
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )

index = pc.Index(index_name)

#download files and split documents
download_files(files_list, folder_name, headers_dic)
docs_after_split = split_documents(folder_path)

#embedding and storing
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = PineconeVectorStore(index=index, embedding=embeddings)
document_ids = [document_id(document) for document in docs_after_split]
vectorstore.add_documents(docs_after_split, ids=document_ids)
print(f"Upserted {len(docs_after_split)} chunks with stable IDs.")

#LLM
model = init_chat_model("gpt-6.1-sol")

#format prompt
prompt_template = """
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

PROMPT = PromptTemplate(
 template=prompt_template, input_variables=["context", "question"]
)

#retrieve
retriever = vectorstore.as_retriever(search_type="similarity" , search_kwargs={"k" : 6})
retrievalQA = RetrievalQA.from_chain_type(
    llm=model, 
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs = {"prompt" : PROMPT}
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

    source_pills = []
    for source in sources:
        label = html.escape(source)
        source_pills.append(f'<span class="source-pill">{label}</span>')

    return (
        '<div class="source-box">'
        '  <div class="source-label">Sources</div>'
        f'  <div class="source-list">{"".join(source_pills)}</div>'
        '</div>'
    )


def chat_with_model(new_message, _=None):

    if not new_message.strip():
        return "Please enter a message."

    result = retrievalQA.invoke({
        "query": new_message
    })

    answer = result["result"].strip()
    source_documents = result["source_documents"]

    sources = []
    seen_sources = set()
    for doc in source_documents:
        metadata = doc.metadata
        source = metadata.get("source")
        source_name = source_title(source)
        page = metadata.get("page")
        page_number = metadata.get("page_label")
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

    answer_has_no_match = "i couldn't find this information in the documents" in answer.casefold()
    if sources and not answer_has_no_match:
        answer += format_sources(sources)
    return answer



def gradio_chat_app():
    custom_css = """
    .gradio-container {
        background: linear-gradient(180deg, #f8fafc 0%, #eef4ff 100%);
    }
    .source-box {
        margin-top: 1.25rem;
        padding: 0.9rem 1rem;
        border: 1px solid #dfe9ff;
        border-radius: 16px;
        background: linear-gradient(180deg, #f8fbff 0%, #eef5ff 100%);
        box-shadow: 0 8px 20px rgba(59, 130, 246, 0.06);
    }
    .source-label {
        font-size: 0.68rem;
        font-weight: 700;
        letter-spacing: 0.10em;
        text-transform: uppercase;
        color: #64748b;
        margin-bottom: 0.7rem;
    }
    .source-list {
        display: flex;
        flex-wrap: wrap;
        gap: 0.5rem;
    }
    .source-pill {
        display: inline-flex;
        align-items: center;
        padding: 0.45rem 0.8rem;
        border-radius: 999px;
        background: #e0ecff;
        border: 1px solid #bfd7ff;
        color: #1d4ed8;
        font-size: 0.78rem;
        font-weight: 600;
        line-height: 1.2;
    }
    """

    with gr.Blocks(css=custom_css) as app:
        gr.Markdown(
            """
            <div style="padding: 1rem 0 0.5rem;">
                <h1 style="margin: 0; font-size: 2rem; color: #1f2937;">Positive Parenting Assistant</h1>
                <p style="margin: 0.35rem 0 0; color: #475569;">Ask about the parenting documents and I’ll answer using the retrieved sources.</p>
            </div>
            """
        )
        chatbot = gr.ChatInterface(
            fn=chat_with_model,
            textbox=gr.Textbox(
                placeholder="Ask a question about positive parenting..."
            ),
            chatbot=gr.Chatbot(
                buttons=["copy"],
                render_markdown=True,
                layout="bubble",
                avatar_images=(None, None),
            ),
            flagging_mode="manual",
            flagging_options=["Like", "Spam", "Inappropriate", "Other"],
            save_history=True,
        )

    return app


if __name__ == "__main__":
    app = gradio_chat_app()
    app.launch( theme=gr.themes.Base())
