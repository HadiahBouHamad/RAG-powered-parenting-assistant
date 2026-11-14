import os
from pinecone import Pinecone, ServerlessSpec
import numpy as np
import gradio as gr
import urllib.request
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.chains import RetrievalQA
from langchain_classic.prompts import PromptTemplate
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from urllib.request import urlretrieve
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
vectorstore.add_documents(docs_after_split)
print(f"Indexed {len(docs_after_split)} chunks.")

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


def chat_with_model(new_message, _=None):

    if not new_message.strip():
        return "Please enter a message."

    # Ask the RAG chain
    result = retrievalQA.invoke({
        "query": new_message
    })

    answer = result["result"]
    source_documents = result["source_documents"]
    
    sources = []
    seen_sources = set()
    for doc in source_documents:
        metadata = doc.metadata
        source = metadata.get("source")
        source_name = os.path.basename(source) if source else "Unknown source"
        page = metadata.get("page")
        page_number = metadata.get("page_label")
        if page_number is None and isinstance(page, int):
            page_number = page + 1

        citation = (
            f"{source_name} (page {page_number})"
            if page_number is not None
            else source_name
        )
        if citation not in seen_sources:
            sources.append(citation)
            seen_sources.add(citation)

    # if sources:
    #     answer += "\n\n**Sources:**\n" + "\n".join(
    #         f"- {source}" for source in sources
    #     )
    return answer



def gradio_chat_app():

    with gr.Blocks() as app:
        gr.Markdown("# My GPT RAG Chatbot")
        chatbot = gr.ChatInterface(
            fn=chat_with_model,
            flagging_mode="manual",
            flagging_options=["Like", "Spam", "Inappropriate", "Other"],
            save_history=True,
            )
    
    return app

if __name__ == "__main__":
    app = gradio_chat_app()
    app.launch(theme=gr.themes.Base())
