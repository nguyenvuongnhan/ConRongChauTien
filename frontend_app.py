# -*- coding: utf-8 -*-
"""
Frontend giao diện cho Chatbot Dược liệu Cổ truyền
Sử dụng Streamlit để tạo giao diện và gọi API từ backend.
"""

import streamlit as st
import requests
import time

# --- Cấu hình giao diện ---
st.set_page_config(page_title="Chatbot Dược liệu", page_icon="🌿", layout="wide")

st.title("🌿 Chatbot Dược liệu Cổ truyền")
st.caption("Giao diện hỏi đáp, được cung cấp bởi FastAPI backend.")

# --- Sidebar để nhập API Token ---
with st.sidebar:
    st.header("Cấu hình")
    st.markdown(
        "Để sử dụng, bạn cần có [Hugging Face API Token](https://https://huggingface.co/settings/tokens)."
    )
    # Dùng session_state để lưu API key, tránh phải nhập lại
    if 'huggingface_api_key' not in st.session_state:
        st.session_state.huggingface_api_key = ''

    huggingface_api_key = st.text_input(
        "Nhập Hugging Face API Token của bạn:",
        type="password",
        value=st.session_state.huggingface_api_key
    )
    if huggingface_api_key:
        st.session_state.huggingface_api_key = huggingface_api_key
        st.success("Đã lưu API token!")
    
# --- Địa chỉ của Backend API ---
BACKEND_URL = "http://127.0.0.1:8000/chat"

# --- Khởi tạo lịch sử chat ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Hiển thị các tin nhắn đã có ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- Xử lý input của người dùng ---
# Chỉ cho phép chat khi đã nhập API key
chat_enabled = bool(st.session_state.huggingface_api_key)
if prompt := st.chat_input("Hỏi tôi về một vị thuốc...", disabled=not chat_enabled):
    # Hiển thị tin nhắn của người dùng
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Gửi yêu cầu đến backend và hiển thị câu trả lời
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response_content = ""
        
        try:
            with st.spinner("Đang suy nghĩ..."):
                # Tạo payload chứa cả câu hỏi và API token
                payload = {
                    "query": prompt,
                    "api_token": st.session_state.huggingface_api_key
                }
                
                # Gọi API từ backend
                response = requests.post(BACKEND_URL, json=payload)
                
                # Kiểm tra lỗi từ backend
                if response.status_code != 200:
                    error_detail = response.json().get("detail", "Lỗi không xác định")
                    st.error(f"Lỗi từ backend (Code: {response.status_code}): {error_detail}")
                    full_response_content = f"Rất tiếc, đã có lỗi từ backend: {error_detail}"
                else:
                    data = response.json()
                    result_text = data.get("result", "Xin lỗi, tôi không tìm thấy câu trả lời.")
                    source_documents = data.get("source_documents", [])

                    # Hiệu ứng gõ chữ
                    for chunk in result_text.split():
                        full_response_content += chunk + " "
                        time.sleep(0.05)
                        message_placeholder.markdown(full_response_content + "▌")
                    message_placeholder.markdown(full_response_content)

                    # Hiển thị nguồn trích dẫn
                    if source_documents:
                        with st.expander("Xem các nguồn trích dẫn"):
                            for doc in source_documents:
                                st.info(doc.get("page_content", ""))

        except requests.exceptions.RequestException as e:
            st.error(f"Lỗi kết nối đến backend: {e}")
            full_response_content = "Rất tiếc, không thể kết nối đến máy chủ chatbot."
            message_placeholder.markdown(full_response_content)
            
    # Lưu câu trả lời của bot vào lịch sử
    st.session_state.messages.append({"role": "assistant", "content": full_response_content})

if not chat_enabled:
    st.info("Vui lòng nhập Hugging Face API Token của bạn vào thanh bên để bắt đầu trò chuyện.")