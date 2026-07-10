import streamlit as st
import requests

st.set_page_config(page_title="Self-Corrective RAG Explorer", page_icon="🤖", layout="wide")
st.title("🤖 Self-Corrective LangGraph RAG Agent")

API_URL = "http://127.0.0.1:8000/api/v1/query"

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "context" in msg and msg["context"]:
            with st.expander("View Grounding Sources"):
                for idx, doc in enumerate(msg["context"]):
                    st.caption(f"**Source {idx+1} Chunk:**")
                    st.text(doc["page_content"])

if prompt := st.chat_input("Ask the RAG Pipeline a technical specification question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.status("Executing graph nodes...", expanded=True) as status:
            try:
                response = requests.post(API_URL, json={"question": prompt}, timeout=60)
                if response.status_code == 200:
                    data = response.json()
                    gen_text = data["generation"]
                    sources = data["context"]
                    retries = data["retry_count"]
                    
                    if retries > 0:
                        status.update(label=f"⚠️ Self-Corrected: Resolved after {retries} loops!", state="complete")
                    else:
                        status.update(label="✅ Guardrail Checked: Clean execution.", state="complete")
                else:
                    st.error(f"Backend API Error: {response.text}")
                    gen_text, sources, retries = "Execution failed.", [], 0
                    status.update(label="💥 Execution broken.", state="error")
            except Exception as e:
                st.error(f"Connection failure: {str(e)}")
                gen_text, sources, retries = "Could not reach backend.", [], 0
                status.update(label="💥 Connection failed.", state="error")

        st.markdown(gen_text)
        if sources:
            with st.expander("View Grounding Sources"):
                for idx, doc in enumerate(sources):
                    st.caption(f"**Source {idx+1} Chunk:**")
                    st.text(doc["page_content"])

        st.session_state.messages.append({
            "role": "assistant",
            "content": gen_text,
            "context": sources
        })