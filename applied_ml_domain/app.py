import streamlit as st
from rag_backend import rag_simple, rag_retriever, llm

# Page config
st.set_page_config(
    page_title="PDF Question Answering System",
    page_icon="📄",
    layout="wide"
)

# Title
st.title("📄 PDF Question Answering System")

st.markdown("---")

# Input box
query = st.text_input(
    "Ask a question:",
    placeholder="Example: How does self-attention differ from recurrence?"
)

# Ask button
if st.button("Ask"):

    if not query.strip():
        st.warning("Please enter a question.")
    
    else:

        with st.spinner("Searching documents and generating answer..."):

            response = rag_simple(
                query=query,
                retriever=rag_retriever,
                llm=llm,
                top_k=5
            )

        st.markdown("---")

        st.subheader("Answer")

        st.write(response)

        st.markdown("---")