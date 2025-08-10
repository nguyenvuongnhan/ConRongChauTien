import json
import torch
import re
import time
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Any, Union
from contextlib import asynccontextmanager

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain.schema import Document, StrOutputParser
from langchain_ollama.llms import OllamaLLM

from sentence_transformers.cross_encoder import CrossEncoder

from langgraph.graph import StateGraph, END, START
from typing_extensions import TypedDict


class ChatRequest(BaseModel):
    query: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 Server đang khởi động và chuẩn bị dữ liệu...")
    load_and_prepare_rag()
    yield
    print("...Server shutting down...")


app = FastAPI(
    title="API Chatbot Y học Cổ truyền",
    description="API cho chatbot Y học Cổ truyền.",
    version="3.4.0",
    lifespan=lifespan
)

retriever = None
reranker = None
rag_app = None


class GraphState(TypedDict):
    question: str
    generation: str
    documents: Union[List[Document], List[dict]]
    llm: Any


def retrieve_node(state):
    documents = retriever.invoke(state["question"])
    print(f"---NODE: RETRIEVE (Completed)---")
    print(f"  Retrieved {len(documents)} document chunks.")
    return {"documents": documents}


def grade_documents_node(state):
    question = state["question"]
    documents = state["documents"]
    llm = state["llm"]

    system = """You are a document relevance grader for a chatbot that assists in Vietnamese traditional medicine. Your goal is to filter out document that is not relevant to the user's question.

    Instructions:
    1. A document is considered RELEVANT ('y') if it contains information that is relevant to the user's question, even if it does not cover every single detail in the question.
    2. A document is NOT RELEVANT ('n') only if it is completely off-topic.
    3. Your job is to filter out entirely irrelevant documents, not to answer the question. Do not be overly strict.
    
    Conclude with a single character: 'y' if relevant or 'n' if not relevant. /no_think"""
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system),
        ("human", "Document:\n\n{document}\n\nQuestion: {question}"),
    ])
    grader = prompt | llm | StrOutputParser()
    
    filtered_docs = []
    grading_results_log = []

    for d in documents:
        doc_content = d.metadata.get("source_record") 
        if not doc_content:
            continue
        response_str = grader.invoke({"question": question, "document": doc_content})
        decision_part = re.sub(r"<think>.*?</think>", "", response_str, flags=re.DOTALL).strip().lower()
        final_word = decision_part.split()[-1] if decision_part.split() else ""
        grade = "no"
        if "y" in final_word:
            grade = "yes"
            filtered_docs.append(d)
        doc_name = doc_content.get('name', 'Unknown')[:40]
        grading_results_log.append(f"  - Doc: '{doc_name}...' -> Grade: {grade.upper()}")

    print("---NODE: GRADE DOCUMENTS (Completed)---")
    print("\n".join(grading_results_log))
    
    return {"documents": filtered_docs}


def rerank_documents_node(state):
    print("---NODE: RERANK DOCUMENTS---")
    question = state["question"]
    documents = state["documents"]

    if not documents:
        return {"documents": []}

    # Create pairs of [question, document_content] for the reranker
    pairs = []
    for d in documents:
        # Rerank based on the text content that was embedded
        pairs.append([question, d.page_content])

    # Get scores from the reranker model
    scores = reranker.predict(pairs)

    # Combine documents with their scores
    scored_docs = list(zip(scores, documents))

    # Sort documents by score in descending order
    scored_docs.sort(key=lambda x: x[0], reverse=True)

    # Filter out documents below a relevance threshold and keep the top N
    final_docs = []
    rerank_log = []
    print("  Reranking results:")
    for score, doc in scored_docs:
        doc_name = doc.metadata.get("source_record", {}).get('name', 'Unknown')[:40]
        rerank_log.append(f"  - Doc: '{doc_name}...' -> Score: {score:.4f}")
        if score > 0.3:
            final_docs.append(doc)
        # Limit to the top 8 most relevant documents
        if len(final_docs) >= 8:
            break
    print("\n".join(rerank_log))
    print(f"  Passing {len(final_docs)} documents to the generator.")
    return {"documents": final_docs}


def generate_node(state):
    question = state["question"]
    documents = state["documents"]
    llm = state["llm"]

    source_records = [
        doc.metadata.get("source_record")
        for doc in documents if hasattr(doc, 'metadata') and doc.metadata.get("source_record")
    ]
    
    prompt_text = """You are an assistant in Vietnamese traditional medicine. Your task is to synthesize information from all provided documents to give a comprehensive answer to the user's question.

    Instructions:
    1.  Carefully read the user's question and analyze **every document** provided in the context.
    2.  Identify all herbs from the context that are relevant to the user's question.
    3.  Construct a final answer that begins with a clear introductory sentence.
    4.  Then, present the relevant herbs in a numbered list. For each herb, state its name and clearly explain how the properties and uses of the herbs is helpful. Do not just mechanically extract text.
    5.  Conclude with a summary statement if appropriate.
    6.  Do not make up information. Base your entire answer ONLY on the provided context.
    7.  Your final, synthesized answer must be in **Vietnamese**.

    Context:
    {context}

    Question: {question}
    Synthesized Answer (in Vietnamese):"""

    prompt_template = PromptTemplate.from_template(prompt_text)
    rag_chain = prompt_template | llm | StrOutputParser()
    generation = rag_chain.invoke({"context": source_records, "question": question})
    print("---NODE: GENERATE (Completed)---")
    
    return {"generation": generation, "documents": source_records}


def load_and_prepare_rag():
    global retriever, rag_app, reranker
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"--- Using device: {device} ---")

    documents = []
    with open('./merged_data.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    for record in data:
        page_content = (
            f"Tên vị thuốc: {record.get('name', '')}. "
            f"Chi tiết: {record.get('detail', '')}. "
            f"Tóm tắt: {record.get('summaried', '')}"
        )
        doc = Document(page_content=page_content, metadata={'source_record': record})
        documents.append(doc)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=100)
    chunks = text_splitter.split_documents(documents)
    model_kwargs = {"device": device, "trust_remote_code": True}
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs=model_kwargs
    )
    vectorstore = FAISS.from_documents(chunks, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 30})
    print("✅ Retriever đã sẵn sàng!")

    reranker = CrossEncoder('BAAI/bge-reranker-v2-m3', max_length=8192, device=device, trust_remote_code=True)
    print("✅ Reranker đã sẵn sàng!")

    workflow = StateGraph(GraphState)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("grade_documents", grade_documents_node)
    workflow.add_node("rerank_documents", rerank_documents_node)
    workflow.add_node("generate", generate_node)
    workflow.add_edge(START, "retrieve")
    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_edge("grade_documents", "rerank_documents")
    # workflow.add_edge("retrieve", "rerank_documents")
    workflow.add_edge("rerank_documents", "generate")
    workflow.add_edge("generate", END)
    rag_app = workflow.compile()
    print("✅ Graph đã được biên dịch và sẵn sàng!")


@app.post("/chat")
async def chat_with_bot(request: ChatRequest):
    if not retriever or not rag_app:
        raise HTTPException(status_code=503, detail="Hệ thống chưa sẵn sàng.")
    llm = OllamaLLM(model="qwen3:8b", num_ctx=10000)
    try:
        inputs = {"question": request.query, "llm": llm}
        final_state = rag_app.invoke(inputs)
        final_docs_used = final_state.get("documents", [])
        return {
            "generation": final_state.get('generation', "Không thể tạo câu trả lời."),
            "documents": final_docs_used
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi trong quá trình xử lý của graph: {e}")


if __name__ == "__main__":
    print("--- CHẠY THỬ NGHIỆM BACKEND ---")
    load_and_prepare_rag()
    llm = OllamaLLM(model="qwen3:8b", num_ctx=10000, reasoning=None)
    test_query = "Tác dụng chính của SINH ĐỊA HOÀNG là gì và ai không nên dùng?"
    print(f"\n❓ Câu hỏi thử nghiệm: {test_query}")
    inputs = {"question": test_query, "llm": llm}
    print("\n\n--- QUÁ TRÌNH XỬ LÝ CỦA GRAPH ---")
    final_state = None
    start_time = time.perf_counter()
    for step in rag_app.stream(inputs):
        node_name = list(step.keys())[0]
        state_after_node = step[node_name]
        print(f"\n--- Đã hoàn thành Node: {node_name} ---")
        final_state = state_after_node
    end_time = time.perf_counter()
    print("\n\n--- KẾT QUẢ CUỐI CÙNG ---")
    print("\n💬 Câu trả lời của Bot:")
    print(final_state.get('generation', "Không có câu trả lời."))
    print("\n📚 Các nguồn tài liệu đã qua thẩm định được sử dụng:")
    final_source_records = final_state.get("documents", [])
    if final_source_records:
        for i, record in enumerate(final_source_records):
            print(f"--- Nguồn {i+1} ---")
            print(json.dumps(record, indent=2, ensure_ascii=False))
            print("-" * 20)
    else:
        print("Không có nguồn tài liệu nào được sử dụng.")
    elapsed_time = end_time - start_time
    print(f"Execution time: {elapsed_time:.6f} seconds")
