# File code dùng để khởi tạo API.
from huggingface_hub import login
login(token="YOUR TOKEN")

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from langserve import add_routes

from src.base.llm_model import get_hf_llm
from src.rag.main import build_rag_chain, InputQA, OutputQA

llm = get_hf_llm(temperature=0.9)
genai_docs = "data_source\\generative_ai"

# --------------- Chain ---------------

genai_chain = build_rag_chain(llm, data_dir=genai_docs, data_type="pdf")

# --------------- App - FastAPI --------------- 

app = FastAPI(
    title="Langchain Server",
    version="1.0",
    description="A Simple API server using Langchain's Runnable interfaces",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins = ["*"],
    allow_credentials = True,
    allow_methods = ["*"],
    allow_headers = ["*"],
    expose_headers = ["*"],
)

# --------------- Routes - FastAPI --------------- 

@app.get("/check")
async def check():
    return {"status": "ok"}

@app.post("/generative_ai", response_model=OutputQA)
async def generative_ai(inputs: InputQA):
    answer = genai_chain.invoke(inputs.question)
    return {"answer": answer}

# --------------- Langserver Routes - FastAPI --------------- 

add_routes(app,
           genai_chain,
           playground_type="default",
           path="/generative_ai"
           )

# uvicorn src.app:app --host "0.0.0.0" --port 8080 --reload
