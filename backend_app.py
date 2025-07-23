# -*- coding: utf-8 -*-
"""
Backend API cho Chatbot Dược liệu Cổ truyền (Kiến trúc CRAG - Hugging Face)
Sử dụng FastAPI và LangGraph.
"""
import os
from fastapi import FastAPI, HTTPException
# *** SỬA LỖI: Sửa 'pantic' thành 'pydantic' ***
from pydantic import BaseModel, Field
from typing import List, Literal, Any

# LangChain components
from langchain_community.document_loaders import JSONLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEndpoint
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain.schema import Document, StrOutputParser

# LangGraph components
from langgraph.graph import StateGraph, END, START
from typing_extensions import TypedDict

# --- Khởi tạo ứng dụng FastAPI ---
app = FastAPI(
    title="API Chatbot Dược liệu (CRAG - Hugging Face)",
    description="API cho chatbot nâng cao chỉ sử dụng Hugging Face.",
    version="2.4.0",
)

# --- Định nghĩa model cho request và response ---
class ChatRequest(BaseModel):
    query: str
    huggingface_api_key: str

class ChatResponse(BaseModel):
    generation: str
    documents: List[str]

# --- Biến toàn cục để lưu trữ Retriever ---
retriever = None

def load_retriever():
    global retriever
    jq_schema = '.[] | "Tên vị thuốc: " + .name + ". Chi tiết: " + .detail + ". Tóm tắt: " + .summaried'
    loader = JSONLoader(file_path='./merged_data.json', jq_schema=jq_schema, text_content=True)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(documents)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    print("✅ Retriever đã sẵn sàng!")

@app.on_event("startup")
async def startup_event():
    print("🚀 Server đang khởi động và chuẩn bị dữ liệu...")
    load_retriever()

# --- Định nghĩa State và các Node của Graph ---
class GraphState(TypedDict):
    question: str
    generation: str
    documents: List[Document]
    # *** CẬP NHẬT: Quay lại quản lý 1 mô hình LLM duy nhất ***
    llm: Any

def retrieve_node(state):
    print("---NODE: RETRIEVE---")
    documents = retriever.invoke(state["question"])
    return {"documents": documents}

def grade_documents_node(state):
    print("---NODE: GRADE DOCUMENTS---")
    question = state["question"]
    documents = state["documents"]
    llm = state["llm"]

    system = "Bạn là người đánh giá mức độ liên quan của tài liệu với câu hỏi. Chỉ trả lời 'yes' nếu tài liệu liên quan, ngược lại trả lời 'no'."
    prompt = ChatPromptTemplate.from_messages([
        ("system", system),
        ("human", "Tài liệu:\n\n{document}\n\nCâu hỏi: {question}"),
    ])
    grader = prompt | llm | StrOutputParser()
    
    filtered_docs = []
    for d in documents:
        score = grader.invoke({"question": question, "document": d.page_content})
        grade = score.strip().lower()
        if 'yes' in grade:
            print("---GRADE: RELEVANT---")
            filtered_docs.append(d)
        else:
            print("---GRADE: NOT RELEVANT---")
    return {"documents": filtered_docs}

def transform_query_node(state):
    print("---NODE: TRANSFORM QUERY---")
    question = state["question"]
    llm = state["llm"]

    system = "Bạn là người viết lại câu hỏi để tối ưu hóa cho tìm kiếm web."
    prompt = ChatPromptTemplate.from_messages([
        ("system", system),
        ("human", "Câu hỏi ban đầu: {question}\n\nHãy tạo ra một câu hỏi cải tiến:"),
    ])
    rewriter = prompt | llm | StrOutputParser()
    better_question = rewriter.invoke({"question": question})
    return {"question": better_question}

def web_search_node(state):
    print("---NODE: WEB SEARCH---")
    question = state["question"]
    documents = state["documents"]
    
    web_search_tool = DuckDuckGoSearchRun()
    web_results = web_search_tool.run(question)
    web_results_doc = Document(page_content=web_results)
    if documents is None:
        documents = []
    documents.append(web_results_doc)
    return {"documents": documents}

def generate_node(state):
    print("---NODE: GENERATE---")
    question = state["question"]
    documents = state["documents"]
    llm = state["llm"]
    
    prompt_template = PromptTemplate.from_template(
        "Dựa vào ngữ cảnh sau đây để trả lời câu hỏi bằng tiếng Việt.\n\nNgữ cảnh: {context}\n\nCâu hỏi: {question}\n\nCâu trả lời:"
    )
    rag_chain = prompt_template | llm | StrOutputParser()
    generation = rag_chain.invoke({"context": documents, "question": question})
    return {"generation": generation}

def decide_to_generate_edge(state):
    print("---EDGE: DECIDE TO GENERATE---")
    if not state["documents"]:
        return "transform_query"
    else:
        return "generate"

# --- API Endpoint để chat ---
@app.post("/chat")
async def chat_with_bot(request: ChatRequest):
    if not retriever:
        raise HTTPException(status_code=503, detail="Hệ thống chưa sẵn sàng.")
    
    # *** CẬP NHẬT: Khởi tạo 1 mô hình LLM duy nhất ***
    llm = HuggingFaceEndpoint(
        repo_id="meta-llama/Llama-2-70b-hf",
        temperature=0.1,
        max_new_tokens=1024,
        repetition_penalty=1.2,
        huggingfacehub_api_token=request.huggingface_api_key
    )

    # Xây dựng Graph
    workflow = StateGraph(GraphState)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("grade_documents", grade_documents_node)
    workflow.add_node("transform_query", transform_query_node)
    workflow.add_node("web_search", web_search_node)
    workflow.add_node("generate", generate_node)
    
    workflow.add_edge(START, "retrieve")
    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_conditional_edges("grade_documents", decide_to_generate_edge, {
        "transform_query": "transform_query",
        "generate": "generate"
    })
    workflow.add_edge("transform_query", "web_search")
    workflow.add_edge("web_search", "generate")
    workflow.add_edge("generate", END)

    crag_app = workflow.compile()
    
    try:
        # *** CẬP NHẬT: Truyền 1 LLM duy nhất vào state ban đầu ***
        inputs = {
            "question": request.query, 
            "llm": llm
        }
        final_state = crag_app.invoke(inputs)
        
        doc_contents = [doc.page_content for doc in final_state.get('documents', [])]
        
        return {
            "generation": final_state.get('generation', "Không thể tạo câu trả lời."),
            "documents": doc_contents
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi trong quá trình xử lý của graph: {e}")
