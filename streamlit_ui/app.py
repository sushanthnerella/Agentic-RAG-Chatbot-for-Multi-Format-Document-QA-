import streamlit as st
import requests
import uuid
import os

# --- Configuration ---
BACKEND_URL = "http://127.0.0.1:8000"

# --- Session State Initialization ---
def initialize_session_state():
    """Initializes the session state variables if they don't exist."""
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "uploaded_files" not in st.session_state:
        st.session_state.uploaded_files = []

# --- Main App Logic ---
def main():
    """Main function to run the Streamlit application."""
    st.set_page_config(page_title="Agentic RAG Chatbot", layout="wide")
    st.title("📄 Agentic RAG Chatbot")
    
    initialize_session_state()

    # --- Sidebar for File Upload ---
    with st.sidebar:
        st.header("Upload Documents")
        st.markdown("""
        Upload your documents here. The chatbot will use these documents to answer your questions. 
        Supported formats: PDF, TXT, DOCX, CSV, PPTX.
        """)
        
        uploaded_files = st.file_uploader(
            "Choose files", accept_multiple_files=True, type=['pdf', 'txt', 'docx', 'csv', 'pptx']
        )

        if uploaded_files and st.button("Process Documents"):
            with st.spinner("Processing documents... This may take a moment."):
                files_to_upload = []
                for file in uploaded_files:
                    # To send a file to the backend, we need to provide it as a tuple
                    # (filename, file_content, file_mimetype)
                    files_to_upload.append(("files", (file.name, file.getvalue(), file.type)))
                
                try:
                    response = requests.post(
                        f"{BACKEND_URL}/upload?session_id={st.session_state.session_id}",
                        files=files_to_upload
                    )
                    if response.status_code == 200:
                        st.session_state.uploaded_files = response.json().get("filenames", [])
                        st.success("Documents processed successfully!")
                    else:
                        st.error(f"Error processing documents: {response.text}")
                except requests.exceptions.RequestException as e:
                    st.error(f"Could not connect to backend: {e}")

        st.divider()
        st.header("Processed Files")
        if st.session_state.uploaded_files:
            for filename in st.session_state.uploaded_files:
                st.info(f"✅ {filename}")
        else:
            st.info("No documents have been processed for this session yet.")

    # --- Chat Interface ---
    st.header("Chat with your Documents")

    # Display existing chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "sources" in message and message["sources"]:
                with st.expander("View Sources"):
                    for source in message["sources"]:
                        st.info(source)

    # Handle new chat input
    if prompt := st.chat_input("Ask a question about your documents..."):
        if not st.session_state.uploaded_files:
            st.warning("Please upload and process at least one document before asking a question.")
        else:
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # Get response from the backend
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                message_placeholder.markdown("Thinking...")
                
                try:
                    chat_payload = {
                        "query": prompt,
                        "session_id": st.session_state.session_id
                    }
                    response = requests.post(f"{BACKEND_URL}/chat", json=chat_payload)
                    
                    if response.status_code == 200:
                        assistant_response = response.json()
                        answer = assistant_response.get("answer", "No answer found.")
                        sources = assistant_response.get("sources", [])
                        
                        message_placeholder.markdown(answer)
                        if sources:
                            with st.expander("View Sources"):
                                for source in sources:
                                    st.info(source)
                        
                        # Add assistant response to chat history
                        st.session_state.messages.append({
                            "role": "assistant", 
                            "content": answer, 
                            "sources": sources
                        })
                    else:
                        error_msg = f"Error from backend: {response.text}"
                        message_placeholder.error(error_msg)
                        st.session_state.messages.append({"role": "assistant", "content": error_msg})

                except requests.exceptions.RequestException as e:
                    error_msg = f"Could not connect to backend: {e}"
                    message_placeholder.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})


if __name__ == "__main__":
    main()
