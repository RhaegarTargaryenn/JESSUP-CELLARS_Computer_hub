import os
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.llms import GooglePalm
from dotenv import load_dotenv
from langchain.document_loaders.csv_loader import CSVLoader

# Load environment variables from .env file
load_dotenv()

# Initialize the Google Palm LLM model
llm = GooglePalm(google_api_key=os.environ["GOOGLE_API_KEY"], temperature=0.1)

# Initialize instructor embeddings using the Hugging Face model
instructor_embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-large")

# Define the FAISS index directory path
vectordb_dir_path = "faiss_index"

def create_vector_db():
    try:
        # Load data from CSV file
        print("Loading data from CSV file...")
        loader = CSVLoader(file_path='Expanded_Jessup_Cellars_QA.csv', source_column="Question")
        data = loader.load()
        print(f"Data loaded: {len(data)} documents")

        # Create a FAISS instance for vector database from 'data'
        print("Creating FAISS vector database...")
        vectordb = FAISS.from_documents(documents=data, embedding=instructor_embeddings)

        # Ensure the directory exists
        os.makedirs(vectordb_dir_path, exist_ok=True)

        # Save vector database locally
        vectordb.save_local(vectordb_dir_path)
        print(f"FAISS index saved to {vectordb_dir_path}")
    except Exception as e:
        print(f"Error creating FAISS vector database: {e}")

def get_qa_chain():
    try:
        # Load the vector database from the local folder
        print("Loading FAISS vector database...")
        vectordb = FAISS.load_local(vectordb_dir_path, instructor_embeddings)
        print("FAISS vector database loaded")

        # Create a retriever for querying the vector database
        retriever = vectordb.as_retriever(score_threshold=0.7)

        # Define the prompt template
        prompt_template = """Given the following context and a question, generate an answer based on this context only.
        In the answer try to provide as much text as possible from "response" section in the source document context without making much changes.
        If the answer is not found in the context, kindly state "I don't know." Don't try to make up an answer.

        CONTEXT: {context}

        QUESTION: {question}"""

        PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

        # Create the QA chain
        chain = RetrievalQA.from_chain_type(llm=llm,
                                            chain_type="stuff",
                                            retriever=retriever,
                                            input_key="query",
                                            return_source_documents=True,
                                            chain_type_kwargs={"prompt": PROMPT})

        return chain
    except Exception as e:
        print(f"Error loading FAISS index: {e}")
        raise

if __name__ == "__main__":
    # Path to check if the index file exists
    vectordb_file_path = os.path.join(vectordb_dir_path, "index.faiss")

    if not os.path.exists(vectordb_file_path):
        print("FAISS index file does not exist, creating a new one...")
        create_vector_db()
    else:
        print("FAISS index file already exists")

    try:
        chain = get_qa_chain()
        response = chain("When was Jessup Cellars first opened?")
        print(response)
    except Exception as e:
        print(f"Error during QA chain execution: {e}")
