import streamlit as st
import requests

st.set_page_config(page_title="Self-Corrective RAG Explorer", page_icon="🤖", layout="wide")
st.title("🤖 Self-Corrective LangGraph RAG Agent")
st.caption("Production hybrid search framework equipped with real-time hallucination evaluation guardrails.")

API_URL = "http://127.0.0.1:8000/api/v1/query"

# Maintain session-level state history across page refreshes
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous conversation elements
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "context" in msg and msg["context"]:
            with st.expander("View Grounding Sources"):
                for idx, doc in enumerate(msg["context"]):
                    st.caption(f"**Source {idx+1} Chunk:**")
                    st.text(doc["page_content"])

# User prompt ingestion
if prompt := st.chat_input("Ask the RAG Pipeline a technical specification question..."):
    
    # Render user prompt input
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate assistant container response
    with st.chat_message("assistant"):
        # Display execution status indicator using Streamlit Status blocks
        with st.status("Executing graph nodes (Retrieve -> Re-rank -> Evaluate)...", expanded=True) as status:
            try:
                response = requests.post(API_URL, json={"question": prompt}, timeout=60)
                
                if response.status_code == 200:
                    data = response.json()
                    gen_text = data["generation"]
                    sources = data["context"]
                    retries = data["retry_count"]
                    
                    # Update status indicator dynamically based on inner graph loops
                    if retries > 0:
                        status.update(label=f"⚠️ Self-Corrected: Resolved after {retries} query rewrite loops!", state="complete")
                    else:
                        status.update(label="✅ Guardrail Checked: Clean execution on first pass.", state="complete")
                else:
                    st.error(f"Backend API Error: {response.text}")
                    gen_text, sources, retries = "Execution failed.", [], 0
                    status.update(label="💥 Execution broken.", state="error")
            except Exception as e:
                st.error(f"Connection failure: {str(e)}")
                gen_text, sources, retries = "Could not reach backend service.", [], 0
                status.update(label="💥 Connection failed.", state="error")

        # Display final generated answer text
        st.markdown(gen_text)
        
        # Display supporting document context attachments if found
        if sources:
            with st.expander("View Grounding Sources"):
                for idx, doc in enumerate(sources):
                    st.caption(f"**Source {idx+1} Chunk:**")
                    st.text(doc["page_content"])

        # Commit run details to state trace
        st.session_state.messages.append({
            "role": "assistant",
            "content": gen_text,
            "context": sources
        })
