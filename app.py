from gradio_app import launch_gradio_app
from rag import create_chat_handler, create_rag_chain


def main():
    retrieval_qa = create_rag_chain()
    launch_gradio_app(create_chat_handler(retrieval_qa))


if __name__ == "__main__":
    main()
