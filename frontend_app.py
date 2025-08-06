import streamlit as st
import requests
import re

# --- Page Configuration ---
st.set_page_config(page_title="Chatbot Y học Cổ truyền", page_icon="🌿", layout="wide")
st.title("🌿 Chatbot Y học Cổ truyền")

# --- Backend URL ---
BACKEND_URL = "http://127.0.0.1:8000/chat"

# --- Initialize chat history ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Display chat messages from history ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        # Display thinking process if it exists
        if "thinking" in message and message["thinking"]:
            with st.expander("Xem quá trình suy nghĩ của Bot"):
                st.info(message["thinking"])
        
        # Display the main content
        st.markdown(message["content"])

        # Display source documents if they exist
        if "documents" in message and message["documents"]:
            with st.expander("Xem các nguồn trích dẫn"):
                for doc in message["documents"]:
                    # Pretty print the JSON object
                    st.json(doc)

# --- Main chat input ---
if prompt := st.chat_input("Hỏi tôi về một vị thuốc..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        thinking_expander = st.expander("Xem quá trình suy nghĩ của Bot", expanded=False)
        sources_expander = st.expander("Xem các nguồn trích dẫn", expanded=False)
        
        try:
            with st.spinner("Đang suy nghĩ và truy xuất thông tin..."):
                # The payload no longer needs an API key
                payload = {"query": prompt}
                response = requests.post(BACKEND_URL, json=payload)
                
                if response.status_code != 200:
                    error_detail = response.json().get("detail", "Lỗi không xác định")
                    st.error(f"Lỗi từ backend (Code: {response.status_code}): {error_detail}")
                    final_answer = f"Rất tiếc, đã có lỗi từ backend: {error_detail}"
                    thinking_process = None
                    source_documents = []
                else:
                    data = response.json()
                    full_generation = data.get("generation", "Xin lỗi, tôi không tìm thấy câu trả lời.")
                    source_documents = data.get("documents", [])
                    
                    # --- Parse the thinking process from the response ---
                    think_match = re.search(r"<think>(.*?)</think>", full_generation, re.DOTALL)
                    
                    if think_match:
                        thinking_process = think_match.group(1).strip()
                        # The final answer is everything after the </think> tag
                        final_answer = full_generation.split("</think>", 1)[-1].strip()
                    else:
                        thinking_process = None
                        final_answer = full_generation.strip()

            # --- Display the results ---
            message_placeholder.markdown(final_answer)
            
            if thinking_process:
                with thinking_expander:
                    st.info(thinking_process)
            
            if source_documents:
                with sources_expander:
                    for doc in source_documents:
                        st.json(doc)

            # Add the complete assistant response to chat history for re-rendering
            st.session_state.messages.append({
                "role": "assistant",
                "content": final_answer,
                "thinking": thinking_process,
                "documents": source_documents
            })

        except requests.exceptions.RequestException as e:
            st.error(f"Lỗi kết nối đến backend: {e}")
            final_answer = "Rất tiếc, không thể kết nối đến máy chủ chatbot."
            message_placeholder.markdown(final_answer)
            st.session_state.messages.append({"role": "assistant", "content": final_answer})
