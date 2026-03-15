"""
app.py

Streamlit Chatbot Frontend for Healthcare RAG API.
Connects to the FastAPI backend running on localhost:8000.
"""

import streamlit as st
import requests

# API endpoints
API_URL = "http://localhost:8000"
QUERY_ENDPOINT = f"{API_URL}/query"

st.set_page_config(
    page_title="Medical Knowledge Assistant",
    page_icon="⚕️",
    layout="wide"
)

# Header
st.title("⚕️ Medical Knowledge Assistant")
st.markdown(
    "Ask health-related questions. Responses are grounded in **MedlinePlus** topics "
    "and **WHO Clinical Guidelines**."
)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Top sidebar for settings
with st.sidebar:
    st.header("⚙️ Settings")
    dev_mode = st.toggle("Show Evidence Chunks", value=False)
    expand_query = st.toggle("Query Expansion (Broader Search)", value=False)
    
    st.markdown("---")
    st.markdown("### About")
    st.markdown(
        "This assistant uses a Hybrid RAG pipeline (BM25 + Semantic Vector Search) "
        "reranked for precision. It enforces strict citation grounding."
    )

# Display chat history on app rerun
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        
        # Display sources if assistant and available
        if msg["role"] == "assistant":
            if "sources" in msg and msg["sources"]:
                with st.expander("📚 View Sources Citations"):
                    for s in msg["sources"]:
                        st.markdown(f"- {s}")
            
            # Show actual chunk text in dev mode
            if dev_mode and "chunks" in msg and msg["chunks"]:
                with st.expander("🔍 Retrieved & Cited Text Chunks"):
                    for i, chunk in enumerate(msg["chunks"], 1):
                        st.markdown(f"**[{i}] Original Text** (Score: {chunk.get('rerank_score', 'N/A')}):")
                        st.info(chunk["text"])

# Handle user input
if prompt := st.chat_input("Ask a medical question... e.g. What are the symptoms of heart failure?"):
    
    # 1. Add user message to chat state
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # 2. Display user message
    with st.chat_message("user"):
        st.markdown(prompt)

    # 3. Get API Response
    with st.chat_message("assistant"):
        with st.spinner("Searching medical guidelines..."):
            try:
                response = requests.post(
                    QUERY_ENDPOINT,
                    json={
                        "question": prompt,
                        "top_k": 5,
                        "expand_query": expand_query
                    },
                    timeout=120  # generous timeout for local LLMs
                )
                
                if response.status_code == 200:
                    data = response.json()
                    answer = data.get("answer", "No answer provided.")
                    sources = data.get("sources", [])
                    chunks = data.get("chunks", [])
                    
                    # Display the answer
                    st.markdown(answer)
                    
                    if sources:
                        with st.expander("📚 View Sources Citations"):
                            for s in sources:
                                st.markdown(f"- {s}")
                    
                    if dev_mode and chunks:
                        with st.expander("🔍 Retrieved & Cited Text Chunks"):
                            for i, chunk in enumerate(chunks, 1):
                                st.markdown(f"**[{i}] Original Text** (Score: {chunk.get('rerank_score', 'N/A')}):")
                                st.info(chunk["text"])

                    # Save to state
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": answer,
                        "sources": sources,
                        "chunks": chunks
                    })
                    
                else:
                    st.error(f"Error {response.status_code}: {response.text}")
                    st.session_state.messages.append({"role": "assistant", "content": f"API Error {response.status_code}"})
                    
            except requests.exceptions.ConnectionError:
                st.error("Failed to connect to the backend. Is the FastAPI server running on port 8000?")
            except Exception as e:
                st.error(f"An error occurred: {e}")
