#!/usr/bin/env python3
from flask import Flask, request, jsonify
from chat_pdf import ChatPDF  # Import your existing ChatPDF class
import os
import logging

# Initialize Flask app
app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Configuration
PDF_FILE_PATH = "C:\\Users\\FA0555TX\\Desktop\\kaam\\local-assistant-examples\\simple-rag\\Data.pdf"

# Initialize ChatPDF
chat_pdf = ChatPDF()

# Load the PDF file once at startup
if os.path.exists(PDF_FILE_PATH):
    chat_pdf.ingest(PDF_FILE_PATH)
else:
    raise FileNotFoundError(f"PDF file not found at {PDF_FILE_PATH}")

# API Endpoints
@app.route("/ask", methods=["POST"])
def ask():
    """
    Endpoint to ask a question to the chatbot.
    Expects a JSON payload with a "query" field.
    """
    data = request.get_json()
    if not data or "query" not in data:
        return jsonify({"error": "Missing 'query' in request body"}), 400

    query = data["query"]
    logger.debug(f"Received query: {query}")

    try:
        # Stream the response and collect it
        response_chunks = []
        for chunk in chat_pdf.ask(query):
            response_chunks.append(chunk)
            logger.debug(f"Streamed chunk: {chunk}")

        response = "".join(response_chunks)
        logger.debug(f"Final response: {response}")

        return jsonify({"response": response})
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route("/status", methods=["GET"])
def status():
    """
    Endpoint to check the status of the API.
    """
    return jsonify({"status": "running", "message": "ChatPDF API is operational"})

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500

if __name__ == "__main__":
    # Run the Flask app
    app.run(host="0.0.0.0", port=5000, debug=True)