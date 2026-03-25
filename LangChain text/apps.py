# -*- coding: utf-8 -*-
"""apps.py
   Flask app to serve Gemini-powered text summarization.
"""

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from summarizer import summarize_text
import traceback

app = Flask(__name__)
CORS(app)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/summarize', methods=['POST'])
def summarize():
    try:
        data = request.get_json()
        user_input = data.get("text", "").strip()
        if not user_input:
            return jsonify({"success": False, "summary": "⚠️ Please enter text to summarize."})

        summary = summarize_text(user_input)
        return jsonify({"success": True, "summary": summary})

    except Exception:
        print(traceback.format_exc())
        return jsonify({"success": False, "summary": "❌ Internal server error occurred."})

if __name__ == '__main__':
    app.run(debug=True)
