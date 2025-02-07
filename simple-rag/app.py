import os
import gradio as gr
from chat_pdf import ChatPDF

# Initialize your chatbot and ingest the PDF
chat_pdf = ChatPDF()
chat_pdf.ingest("Data.pdf")  # Ensure Data.pdf is in the correct location relative to this file

# Define a function that Gradio will call when a user submits a query
def answer_question(query):
    return chat_pdf.ask(query)

# Create the Gradio interface
iface = gr.Interface(
    fn=answer_question,
    inputs=gr.Textbox(lines=2, placeholder="Enter your question here...", label="Question"),
    outputs=gr.Textbox(label="Answer"),
    title="PDF Chatbot",
    description="Ask questions about the PDF document."
)

# Launch the interface
# Using server_name="0.0.0.0" and setting server_port based on the PORT env variable ensures compatibility
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    iface.launch(server_name="0.0.0.0", server_port=port, share=False)
