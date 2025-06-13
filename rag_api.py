from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
import os
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel
from typing import Optional
import tempfile
import uuid

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(title="PDF Chat API", description="Upload a PDF and ask questions about its content")

# In-memory storage for vectorstores (in production, use Redis or a database)
vectorstore_storage = {}

# Pydantic models for request/response
class QuestionRequest(BaseModel):
    question: str
    session_id: str

class ChatResponse(BaseModel):
    answer: str
    session_id: str
    timestamp: str

class HistoryResponse(BaseModel):
    history: list

def save_conversation(question: str, answer: str, pdf_name: str, session_id: str):
    """Save the conversation to a CSV file"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Create a DataFrame with the new conversation
    new_conversation = pd.DataFrame({
        'Timestamp': [timestamp],
        'Session_ID': [session_id],
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

@app.get("/")
async def root():
    return {"message": "PDF Chat API is running. Use /docs to see available endpoints."}

@app.post("/upload-pdf/")
async def upload_pdf(file: UploadFile = File(...)):
    """Upload and process a PDF file"""
    
    # Validate file type
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")
    
    # Generate session ID
    session_id = str(uuid.uuid4())
    
    try:
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        # Load PDF
        loader = PyPDFLoader(temp_file_path)
        pages = loader.load()
        
        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        splits = text_splitter.split_documents(pages)

        # Create embeddings and vector store
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        vectorstore = FAISS.from_documents(splits, embeddings)
        
        # Store vectorstore with session ID and PDF name
        vectorstore_storage[session_id] = {
            'vectorstore': vectorstore,
            'pdf_name': file.filename
        }
        
        # Clean up temporary file
        os.unlink(temp_file_path)
        
        return {
            "message": "PDF processed successfully",
            "session_id": session_id,
            "pdf_name": file.filename
        }
        
    except Exception as e:
        # Clean up temporary file if it exists
        if 'temp_file_path' in locals():
            try:
                os.unlink(temp_file_path)
            except:
                pass
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")

@app.post("/ask-question/", response_model=ChatResponse)
async def ask_question(request: QuestionRequest):
    """Ask a question about the uploaded PDF"""
    
    # Check if session exists
    if request.session_id not in vectorstore_storage:
        raise HTTPException(status_code=404, detail="Session not found. Please upload a PDF first.")
    
    try:
        
        session_data = vectorstore_storage[request.session_id]
        vectorstore = session_data['vectorstore']
        pdf_name = session_data['pdf_name']
        
        # Initialize Gemini Pro
        llm = ChatOpenAI(model="gpt-4.1-nano", temperature=0.7)
        
        # Create prompt template
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful assistant that answers questions based on the provided context.
            Answer the question using only the context provided. If you're unsure or the answer isn't in 
            the context, say "I don't have enough information to answer that question."
            
            Context: {context}"""),
            ("human", "{input}")
        ])
        
        # Create retrieval chain
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 4}
        )
        
        # Create document chain
        document_chain = create_stuff_documents_chain(llm, prompt)
        
        # Create retrieval chain
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        
        # Get response
        response = retrieval_chain.invoke({"input": request.question})
        answer = response["answer"]
        
        # Save conversation to CSV
        save_conversation(request.question, answer, pdf_name, request.session_id)
        
        return ChatResponse(
            answer=answer,
            session_id=request.session_id,
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating response: {str(e)}")

@app.get("/conversation-history/")
async def get_conversation_history(session_id: Optional[str] = None):
    """Get conversation history, optionally filtered by session ID"""
    
    try:
        if not os.path.exists('record.csv'):
            return {"message": "No conversation history found"}
        
        history = pd.read_csv('record.csv')
        
        # Filter by session ID if provided
        if session_id:
            history = history[history['Session_ID'] == session_id]
            if history.empty:
                return {"message": f"No conversation history found for session {session_id}"}
        
        # Convert to dict for JSON response
        history_dict = history.to_dict('records')
        
        return {
            "total_conversations": len(history_dict),
            "history": history_dict
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading conversation history: {str(e)}")

@app.delete("/clear-session/{session_id}")
async def clear_session(session_id: str):
    """Clear a specific session's vectorstore from memory"""
    
    if session_id not in vectorstore_storage:
        raise HTTPException(status_code=404, detail="Session not found")
    
    del vectorstore_storage[session_id]
    
    return {"message": f"Session {session_id} cleared successfully"}

@app.get("/active-sessions/")
async def get_active_sessions():
    """Get list of active sessions"""
    
    sessions = []
    for session_id, data in vectorstore_storage.items():
        sessions.append({
            "session_id": session_id,
            "pdf_name": data['pdf_name']
        })
    
    return {
        "total_active_sessions": len(sessions),
        "sessions": sessions
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)