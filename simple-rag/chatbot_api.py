from flask import Flask, request, jsonify
from chat_pdf import ChatPDF  # Import your chatbot class (make sure it's in the same directory)
from flask_cors import CORS
app = Flask(__name__)
CORS(app)
# Initialize the chatbot instance
chat_pdf = ChatPDF()

# Ingest the PDF document before starting the API
# You can replace "Data.pdf" with the correct path if needed
chat_pdf.ingest("Data.pdf")

@app.route("/ask", methods=["POST"])
def ask():
    """
    Endpoint to ask questions to the chatbot.
    Accepts a JSON payload with a 'query' field.
    """
    # Get the question from the request body
    data = request.get_json()
    query = data.get("query", "")
    
    # If no question is provided, return an error
    if not query:
        return jsonify({"error": "No question provided"}), 400

    # Get the response from the chatbot
    try:
        answer = chat_pdf.ask(query)
        return jsonify({"answer": answer})
    except Exception as e:
        # If there's an error processing the query, return an error message
        return jsonify({"error": f"An error occurred while processing the query: {str(e)}"}), 500

# if __name__ == "__main__":
#     # Run the Flask app
#     # This will start the API on http://0.0.0.0:5000 by default
#     app.run(host="0.0.0.0", port=5000, debug=True)

import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Use Render's PORT or default to 5000
    app.run(host="0.0.0.0", port=port, debug=False)  # Turn debug off in production

