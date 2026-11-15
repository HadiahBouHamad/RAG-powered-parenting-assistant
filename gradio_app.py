from pathlib import Path

import gradio as gr


def gradio_chat_app(chat_function):
    with gr.Blocks() as app:
        gr.Markdown(
            """
            <div style="padding: 1rem 0 0.5rem;">
                <h1 style="margin: 0; font-size: 2rem; color: #1f2937;">Positive Parenting Assistant</h1>
                <p style="margin: 0.35rem 0 0; color: #475569;">Ask about the parenting documents and I’ll answer using the retrieved sources.</p>
            </div>
            """
        )
        gr.ChatInterface(
            fn=chat_function,
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


def launch_gradio_app(chat_function):
    custom_css = Path(__file__).with_name("styles.css").read_text(encoding="utf-8")
    app = gradio_chat_app(chat_function)
    app.launch(css=custom_css, theme=gr.themes.Base())
