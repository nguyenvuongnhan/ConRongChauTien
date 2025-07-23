# -*- coding: utf-8 -*-
"""
Backend API cho Chatbot Dược liệu Cổ truyền (Kiến trúc CRAG - Hugging Face)
Sử dụng FastAPI và LangGraph.
File này cũng có thể được chạy trực tiếp để kiểm tra logic.
"""
import os
from fastapi import FastAPI, HTTPException
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
    version="2.6.0",
)

# --- Định nghĩa model cho request và response ---
class ChatRequest(BaseModel):
    query: str
    huggingface_api_key: str

class ChatResponse(BaseModel):
    generation: str
    documents: List[str]

# --- Biến toàn cục để lưu trữ các thành phần đã được xử lý trước ---
retriever = None
crag_app = None

# --- Định nghĩa State và các Node của Graph ---
class GraphState(TypedDict):
    question: str
    generation: str
    documents: List[Document]
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

def load_and_prepare_rag():
    global retriever, crag_app
    
    jq_schema = '.[] | "Tên vị thuốc: " + .name + ". Chi tiết: " + .detail + ". Tóm tắt: " + .summaried'
    loader = JSONLoader(file_path='./merged_data.json', jq_schema=jq_schema, text_content=True)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(documents)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    print("✅ Retriever đã sẵn sàng!")

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
    print("✅ Graph đã được biên dịch và sẵn sàng!")

@app.on_event("startup")
async def startup_event():
    print("🚀 Server đang khởi động và chuẩn bị dữ liệu...")
    load_and_prepare_rag()

@app.post("/chat")
async def chat_with_bot(request: ChatRequest):
    if not retriever or not crag_app:
        raise HTTPException(status_code=503, detail="Hệ thống chưa sẵn sàng.")
    
    llm = HuggingFaceEndpoint(
        repo_id="meta-llama/Llama-2-70b-hf",
        temperature=0.1,
        max_new_tokens=1024,
        repetition_penalty=1.2,
        huggingfacehub_api_token=request.huggingface_api_key
    )
    
    try:
        inputs = {"question": request.query, "llm": llm}
        final_state = crag_app.invoke(inputs)
        doc_contents = [doc.page_content for doc in final_state.get('documents', [])]
        
        return {
            "generation": final_state.get('generation', "Không thể tạo câu trả lời."),
            "documents": doc_contents
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi trong quá trình xử lý của graph: {e}")

# --- CẬP NHẬT: Thêm hàm chính để chạy thử nghiệm ---
if __name__ == "__main__":
    # Hàm này chỉ chạy khi bạn thực thi file: `python backend_app.py`
    
    print("--- CHẠY THỬ NGHIỆM BACKEND ---")
    
    # 1. Kiểm tra API Key từ biến môi trường
    api_key = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    if not api_key:
        print("\n❌ Lỗi: Vui lòng thiết lập biến môi trường HUGGINGFACEHUB_API_TOKEN trước khi chạy.")
        print("Ví dụ: export HUGGINGFACEHUB_API_TOKEN='hf_your_token_here'")
    else:
        # 2. Tải dữ liệu và biên dịch graph
        load_and_prepare_rag()
        
        # 3. Khởi tạo LLM
        llm = HuggingFaceEndpoint(
            repo_id="meta-llama/Llama-2-70b-hf",
            temperature=0.1,
            max_new_tokens=1024,
            repetition_penalty=1.2,
            huggingfacehub_api_token=api_key
        )

        # 4. Đặt câu hỏi thử nghiệm
        test_query = "vị thuốc nào có tác dụng bổ huyết và cầm máu?"
        print(f"\n❓ Câu hỏi thử nghiệm: {test_query}")

        # 5. Chạy Graph
        inputs = {"question": test_query, "llm": llm}
        final_state = crag_app.invoke(inputs)

        # 6. In kết quả
        print("\n\n--- KẾT QUẢ CUỐI CÙNG ---")
        print("\n💬 Câu trả lời của Bot:")
        print(final_state.get('generation', "Không có câu trả lời."))
        
        print("\n📚 Các nguồn tài liệu được sử dụng:")
        source_docs = final_state.get('documents', [])
        if source_docs:
            for i, doc in enumerate(source_docs):
                print(f"--- Nguồn {i+1} ---")
                print(doc.page_content)
                print("-" * 20)
        else:
            print("Không có nguồn tài liệu nào được sử dụng.")

"""
--- KẾT QUẢ MẪU KHI CHẠY THỬ NGHIỆM (python backend_app.py) ---

--- CHẠY THỬ NGHIỆM BACKEND ---
🚀 Server đang khởi động và chuẩn bị dữ liệu...
✅ Retriever đã sẵn sàng!
✅ Graph đã được biên dịch và sẵn sàng!

❓ Câu hỏi thử nghiệm: vị thuốc nào có tác dụng bổ huyết và cầm máu?
---NODE: RETRIEVE---
---NODE: GRADE DOCUMENTS---
---GRADE: RELEVANT---
---GRADE: RELEVANT---
---GRADE: RELEVANT---
---GRADE: RELEVANT---
---EDGE: DECIDE TO GENERATE---
---NODE: GENERATE---


--- KẾT QUẢ CUỐI CÙNG ---

💬 Câu trả lời của Bot:
Dựa vào ngữ cảnh được cung cấp, các vị thuốc có tác dụng bổ huyết và cầm máu bao gồm:
1.  **A GIAO (Phó tri giao):** Tác dụng chính là Bổ Huyết và Cầm Máu, chữa các chứng xuất huyết và các bệnh do huyết khô.
2.  **BẠCH THƯỢC:** Có khả năng Dưỡng Huyết, Liễm Âm, dùng để nuôi máu.
3.  **ĐAN-SÂM:** Công hiệu được ví như bài Tứ vật thang, chuyên trị các bệnh về huyết.
4.  **HÀ-DIỆP (Lá sen):** Giúp cầm các chứng xuất huyết như thổ huyết, băng huyết.

📚 Các nguồn tài liệu được sử dụng:
--- Nguồn 1 ---
Tên vị thuốc: Phó tri giao, Chân a giao, Lư bì giao, A giao châu, Sao a giao, Thượng a giao, Ô giao, Bì giao. Chi tiết: Thuốc này tên là A GIAO, tác dụng là chữa các chứng xuất huyết (đổ máu cam, tiểu/đại tiện ra huyết, băng huyết), các chứng do huyết khô (đau lưng, kinh nguyệt không đều) và động thai.. Tóm tắt: Tác dụng chính nổi bật là **Bổ Huyết và Cầm Máu**. Nhờ vào tính vị ngọt, tính bình, không độc, A Giao có khả năng tư âm, nhuận táo, chuyên trị các chứng ho khan do phế táo, các loại xuất huyết do huyết nhiệt hoặc hư tổn, và an thai. Không có tác dụng phụ được đề cập, tuy nhiên cần lưu ý: người có tỳ vị hư yếu, ăn uống khó tiêu, hoặc đang bị tiêu chảy thì không nên dùng. Vị thuốc này kỵ (sợ) Đại hoàng.
--------------------
... (các nguồn tài liệu liên quan khác sẽ được liệt kê ở đây) ...
"""
