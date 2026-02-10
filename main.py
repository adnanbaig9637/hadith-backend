# main.py

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import uvicorn
from backend.initial_rag_engine import  query_rag



app = FastAPI(
    title="Hadith RAG Engine",
    description="Retrieve Hadiths and Islamic scholarly guidance with AI",
    version="1.0.0"
)

# Add CORS middleware to allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Frontend URL
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods (GET, POST, OPTIONS, etc.)
    allow_headers=["*"],  # Allow all headers
)

class QueryRequest(BaseModel):
    query: str
    top_k: Optional[int] = 10

class QueryResponse(BaseModel):
    answer: str
    top_k: int

@app.get("/")
async def root():
    return {"message": "Welcome to the Hadith RAG Engine. Use /query endpoint to ask questions."}

@app.post("/query", response_model=QueryResponse)
async def query_hadith(request: QueryRequest):
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    try:
        # Call your existing RAG function
        answer = query_rag(request.query, top_k=request.top_k)
        return QueryResponse(answer=answer, top_k=request.top_k)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8009, reload=True)
