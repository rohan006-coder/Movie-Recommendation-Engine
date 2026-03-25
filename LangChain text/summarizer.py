# -*- coding: utf-8 -*-
"""summarizer.py
   Text summarization using LangChain + Google Gemini API.
"""

import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

# Load environment variables
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

def _local_summarize(input_text: str) -> str:
    """Simple offline fallback summarization."""
    import re

    # Normalize and split into sentences.
    text = input_text.strip()
    if not text:
        return ""

    sentences = re.split(r'(?<=[.!?])\s+', text)
    if len(sentences) <= 2:
        return text

    # Choose the first and most informative sentences by length.
    top = sorted(sentences, key=len, reverse=True)[:2]
    summary = ' '.join(top)
    if len(summary) > 500:
        summary = ' '.join(top[:1])
    return summary

# Initialize Gemini model if API key is provided
if api_key:
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=api_key,
        temperature=0.3
    )
else:
    llm = None

# Define summarization prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful and concise AI assistant that summarizes text clearly."),
    ("human", "Please summarize the following text:\n\n{text}")
])

def summarize_text(input_text: str) -> str:
    """Summarize input text using Gemini API (or local fallback)."""
    if not input_text or not input_text.strip():
        return "⚠️ Please provide valid text to summarize."

    if llm is None:
        return _local_summarize(input_text)

    try:
        formatted_prompt = prompt.format_messages(text=input_text)
        response = llm.invoke(formatted_prompt)
        return response.content.strip()
    except Exception as e:
        print(f"⚠️ Summarization Error: {e}")
        return _local_summarize(input_text)
