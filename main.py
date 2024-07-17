import streamlit as st
from langchain_helper import get_qa_chain, create_vector_db

st.title("Jessup Cellars Q&A 🌱")

# Button to create the knowledge base
if st.button("Create Knowledgebase"):
    create_vector_db()
    st.success("Knowledgebase created successfully!")

# Text input for the question
question = st.text_input("Question: ")

if question:
    try:
        chain = get_qa_chain()
        response = chain(question)
        st.header("Answer")
        st.write(response["result"])

        # Optionally display source documents
        st.header("Source Documents")
        for doc in response["source_documents"]:
            st.write(f"Document: {doc.metadata['source']}")
            st.write(f"Content: {doc.page_content}")
            st.write("---")

    except Exception as e:
        st.error(f"An error occurred: {e}")
