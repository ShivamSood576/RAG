import streamlit as st
import os
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# Page config
st.set_page_config(page_title="PDF Chat App", page_icon="📚")

# Load environment variables
load_dotenv()

# Initialize session state
if 'processed' not in st.session_state:
    st.session_state.processed = False

def save_conversation(question, answer, pdf_name):
    """Save the conversation to a CSV file"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Create a DataFrame with the new conversation
    new_conversation = pd.DataFrame({
        'Timestamp': [timestamp],
        'PDF_Name': [pdf_name],
        'Question': [question],
        'Answer': [answer]
    })
    
    # Define the CSV file path
    csv_file = 'record.csv'
    
    # If file exists, append without header; if not, create with header
    if os.path.exists(csv_file):
        new_conversation.to_csv(csv_file, mode='a', header=False, index=False)
    else:
        new_conversation.to_csv(csv_file, mode='w', header=True, index=False)

# Streamlit UI
st.title("Chat with PDF ")
st.write("Upload a PDF and ask questions about its content!")

# File upload
uploaded_file = st.file_uploader("Choose a PDF file", type=['pdf'])

if uploaded_file is not None:
    pdf_name = uploaded_file.name  # Store PDF name for logging
    
    if not st.session_state.processed:
        try:
            # Save uploaded file temporarily
            with open("temp.pdf", "wb") as f:
                f.write(uploaded_file.getvalue())
            
            # Load PDF
            st.info("Processing PDF... Please wait.")
            loader = PyPDFLoader("temp.pdf")
            pages = loader.load()
            
            # Split text into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            splits = text_splitter.split_documents(pages)

            # Create embeddings and vector store
            embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
            vectorstore = FAISS.from_documents(splits, embeddings)
            
            # Save vectorstore in session state
            st.session_state.vectorstore = vectorstore
            st.session_state.processed = True
            
            # Remove temporary file
            os.remove("temp.pdf")
            
            st.success("PDF processed successfully!")
            
        except Exception as e:
            st.error(f"Error processing PDF: {str(e)}")
            st.stop()

    # Chat interface
    if st.session_state.processed:
        # Initialize Gemini Pro with the correct model name
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.7)
        
        # Create prompt template
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful assistant that answers questions based on the provided context.
            Answer the question using only the context provided. If you're unsure or the answer isn't in 
            the context, say "I don't have enough information to answer that question."
            
            Context: {context}"""),
            ("human", "{input}")
        ])
        
        # Create retrieval chain
        retriever = st.session_state.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 4}
        )
        
        # Create document chain
        document_chain = create_stuff_documents_chain(llm, prompt)
        
        # Create retrieval chain
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        
        # Chat input
        question = st.text_input("Ask a question about your PDF:")
        
        if question:
            try:
                # Get response
                response = retrieval_chain.invoke({"input": question})
                answer = response["answer"]
                
                # Display response
                st.write("### Answer:")
                st.write(answer)
                
                # Save conversation to CSV
                save_conversation(question, answer, pdf_name)
                
                # Show success message for logging
                st.success("Conversation saved to record.csv")
                
            except Exception as e:
                st.error(f"Error generating response: {str(e)}")

        # Add button to view conversation history
        if st.button("View Conversation History"):
            try:
                if os.path.exists('record.csv'):
                    history = pd.read_csv('record.csv')
                    st.write("### Conversation History")
                    st.dataframe(history)
                else:
                    st.info("No conversation history yet.")
            except Exception as e:
                st.error(f"Error loading conversation history: {str(e)}")

# Display instructions if no file is uploaded
else:
    st.info("👆 Please upload a PDF file to begin.")

# Add some spacing
st.write("")

# Footer
st.markdown("---")
st.markdown("Built with Streamlit, LangChain")