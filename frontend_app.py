# -*- coding: utf-8 -*-
"""
Frontend giao diện cho Chatbot Dược liệu Cổ truyền
Sử dụng Streamlit để tạo giao diện và gọi API từ backend.
"""
import streamlit as st
import requests
import time

st.set_page_config(page_title="Chatbot Dược liệu", page_icon="🌿", layout="wide")
st.title("🌿 Chatbot Dược liệu Cổ truyền (CRAG - Hugging Face)")
st.caption("Giao diện hỏi đáp nâng cao, được cung cấp bởi FastAPI backend.")

with st.sidebar:
    st.header("Cấu hình API")
    st.markdown("Bạn chỉ cần [Hugging Face API Token](https://huggingface.co/settings/tokens) để sử dụng.")
    
    if 'huggingface_api_key' not in st.session_state:
        st.session_state.huggingface_api_key = ''

    huggingface_key = st.text_input("Nhập Hugging Face API Key:", type="password", value=st.session_state.huggingface_api_key)

    if huggingface_key:
        st.session_state.huggingface_api_key = huggingface_key
        st.success("Đã lưu API key!")

BACKEND_URL = "http://127.0.0.1:8000/chat"

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

chat_enabled = bool(st.session_state.huggingface_api_key)
if prompt := st.chat_input("Hỏi tôi về một vị thuốc...", disabled=not chat_enabled):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response_content = ""
        
        try:
            with st.spinner("Đang suy nghĩ và truy xuất thông tin..."):
                payload = {
                    "query": prompt,
                    "huggingface_api_key": st.session_state.huggingface_api_key,
                }
                response = requests.post(BACKEND_URL, json=payload)
                
                if response.status_code != 200:
                    error_detail = response.json().get("detail", "Lỗi không xác định")
                    st.error(f"Lỗi từ backend (Code: {response.status_code}): {error_detail}")
                    full_response_content = f"Rất tiếc, đã có lỗi từ backend: {error_detail}"
                else:
                    data = response.json()
                    result_text = data.get("generation", "Xin lỗi, tôi không tìm thấy câu trả lời.")
                    source_documents = data.get("documents", [])
                    
                    full_response_content = result_text
                    message_placeholder.markdown(full_response_content)

                    if source_documents:
                        with st.expander("Xem các nguồn trích dẫn"):
                            for doc in source_documents:
                                st.info(doc)
        except requests.exceptions.RequestException as e:
            st.error(f"Lỗi kết nối đến backend: {e}")
            full_response_content = "Rất tiếc, không thể kết nối đến máy chủ chatbot."
            message_placeholder.markdown(full_response_content)
            
    st.session_state.messages.append({"role": "assistant", "content": full_response_content})

if not chat_enabled:
    st.info("Vui lòng nhập Hugging Face API Key vào thanh bên để bắt đầu.")
