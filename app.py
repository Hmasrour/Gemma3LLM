import streamlit as st
from document_processor import DocumentProcessor
import os
from dotenv import load_dotenv
import tempfile
from gemma_llm import Gemma3Llm
import warnings

warnings.filterwarnings("ignore")


load_dotenv()
gemma_llm = Gemma3Llm(
    api_key=os.getenv("GEMMA_API_KEY"),
    base_url=os.getenv("GEMMA_API_BASE")
)

# Initialize processor
processor = DocumentProcessor()

# App title
st.title("Report Analyzer with Gemma 3")

# Sidebar for document upload
with st.sidebar:
    st.header("Upload Reports")
    uploaded_files = st.file_uploader(
        "Choose PDF files",
        type="pdf",
        accept_multiple_files=True
    )

    report_type = st.selectbox(
        "Report Type",
        ["ARGUS", "CRU", "Other"]
    )

    report_date = st.date_input("Report Date")

    if st.button("Process Documents"):
        if uploaded_files:
            with st.spinner("Processing documents..."):
                for uploaded_file in uploaded_files:
                    with tempfile.NamedTemporaryFile(delete=False) as tmp:
                        tmp.write(uploaded_file.getvalue())
                        tmp_path = tmp.name

                    metadata = {
                        "report_type": report_type,
                        "date": str(report_date),
                        "filename": uploaded_file.name
                    }

                    num_chunks = processor.process_document(tmp_path, metadata)
                    os.unlink(tmp_path)

                    st.success(f"Processed {uploaded_file.name} into {num_chunks} chunks")
        else:
            st.warning("Please upload at least one document")

# Main chat interface
st.header("Ask Questions About Your Reports")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What would you like to know about the reports?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Retrieve relevant chunks
    collection = processor.client.get_collection("reports")
    query_embedding = processor.embedding_model.encode(prompt).tolist()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=5  # How many results to return(5 most similar results)
    )

    # Build context
    context = "\n\n".join(results["documents"][0])

    # Generate response with Gemma 3
    with st.chat_message("assistant"):
        with st.spinner("Analyzing reports..."):
            response = gemma_llm.generate_response(prompt, context)
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
