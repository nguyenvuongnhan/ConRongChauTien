# -*- coding: utf-8 -*-
"""
Backend API cho Chatbot Dược liệu Cổ truyền
Sử dụng FastAPI để cung cấp endpoint cho việc hỏi đáp.
"""

import os
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from typing import Optional

from langchain_community.document_loaders import JSONLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEndpoint
from langchain.prompts import PromptTemplate

# --- Khởi tạo ứng dụng FastAPI ---
app = FastAPI(
    title="API Chatbot Dược liệu",
    description="API cho phép hỏi đáp dựa trên ngữ liệu y học cổ truyền.",
    version="1.0.0",
)

# --- Định nghĩa model cho request và response ---
class ChatRequest(BaseModel):
    query: str
    api_token: Optional[str] = None

class ChatResponse(BaseModel):
    result: str
    source_documents: list[dict]

# --- Biến toàn cục để lưu trữ Retriever và Prompt ---
retriever = None
prompt_template = None

# --- Hàm tải và chuẩn bị Retriever ---
def load_rag_dependencies():
    """
    Tải dữ liệu, embed, khởi tạo retriever và prompt.
    Hàm này được gọi một lần khi server khởi động.
    """
    global retriever, prompt_template
    
    # 1. Tải dữ liệu từ file JSON
    jq_schema = '.[] | "Tên vị thuốc: " + .name + ". Chi tiết: " + .detail + ". Tóm tắt: " + .summaried'
    loader = JSONLoader(
        file_path='./merged_data.json',
        jq_schema=jq_schema,
        text_content=True
    )
    documents = loader.load()

    # 2. Chia nhỏ văn bản
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(documents)

    # 3. Tạo embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )

    # 4. Lưu vào Vector Store và tạo retriever
    vectorstore = FAISS.from_documents(chunks, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
    # *** SỬA LỖI: Đơn giản hóa mẫu prompt để tránh lặp ***
    template = """
    Dựa vào ngữ cảnh sau đây để trả lời câu hỏi một cách ngắn gọn và chính xác.

    Ngữ cảnh: {context}

    Câu hỏi: {question}

    Câu trả lời:
    """
    prompt_template = PromptTemplate(template=template, input_variables=["context", "question"])
    
    print("✅ Retriever và Prompt đã sẵn sàng!")

# --- Sự kiện khởi động server ---
@app.on_event("startup")
async def startup_event():
    print("🚀 Server đang khởi động và chuẩn bị dữ liệu...")
    load_rag_dependencies()

# --- API Endpoint để chat ---
@app.post("/chat", response_model=ChatResponse)
async def chat_with_bot(request: ChatRequest):
    if not retriever or not prompt_template:
        raise HTTPException(status_code=503, detail="Hệ thống chưa sẵn sàng, vui lòng thử lại sau.")
    
    if not request.api_token:
        raise HTTPException(status_code=400, detail="Hugging Face API Token là bắt buộc.")

    try:
        # Khởi tạo LLM với token được cung cấp trong request
        llm = HuggingFaceEndpoint(
            repo_id="meta-llama/Llama-2-70b-hf",
            temperature=0.1,
            max_new_tokens=1024,
            # *** SỬA LỖI: Thêm tham số chống lặp ***
            repetition_penalty=1.2,
            huggingfacehub_api_token=request.api_token
        )
        
        # Thêm prompt tùy chỉnh vào chuỗi RAG
        chain_type_kwargs = {"prompt": prompt_template}
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs=chain_type_kwargs,
            return_source_documents=True
        )

        response = qa_chain.invoke({"query": request.query})
        
        # Chuyển đổi Document objects thành dict để tương thích JSON
        source_docs = [
            {"page_content": doc.page_content, "metadata": doc.metadata} 
            for doc in response.get('source_documents', [])
        ]
        
        return ChatResponse(
            result=response.get('result', "Xin lỗi, tôi không tìm thấy câu trả lời."),
            source_documents=source_docs
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- Endpoint kiểm tra sức khỏe ---
@app.get("/")
def read_root():
    return {"status": "API Chatbot đang hoạt động!"}
