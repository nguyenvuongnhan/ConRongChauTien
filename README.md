# ConRongChauTien

Hướng dẫn Cài đặt và Chạy Chatbot với Anaconda


**Bước 1: Cài đặt Ollama**

Trước tiên, bạn cần tải và cài đặt ứng dụng Ollama cho hệ điều hành của mình.

    Truy cập trang web chính thức: ollama.com

    Nhấn vào nút "Download" và chọn phiên bản phù hợp với hệ điều hành của bạn (Windows, macOS, hoặc Linux).

    Chạy file cài đặt vừa tải về và làm theo các hướng dẫn. Sau khi cài đặt xong, Ollama sẽ tự động chạy dưới dạng một dịch vụ nền trên máy của bạn.

Để kiểm tra xem Ollama đã được cài đặt thành công chưa, hãy mở cửa sổ dòng lệnh (Terminal, Command Prompt, hoặc PowerShell) và gõ: ollama --version



**Bước 2: Tải Mô Hình (Pull a Model)**

Sau khi cài đặt Ollama, bạn cần tải một mô hình ngôn ngữ lớn (LLM) để sử dụng. Bạn có thể tìm danh sách các mô hình có sẵn tại thư viện của Ollama.

Mở cửa sổ dòng lệnh và sử dụng lệnh ollama pull theo sau là tên mô hình bạn muốn.

Chạy và Tương tác với Mô Hình:
ollama run qwen3:30b


**Bước 3: Chuẩn bị Thư mục Dự án**

Đảm bảo bạn đã có một thư mục chứa đầy đủ 4 file sau:

    backend_app.py

    frontend_app.py

    requirements.txt

    merged_data.json

**Bước 4: Cài đặt Môi trường Anaconda**

    Mở Terminal:

        macOS: Mở ứng dụng Terminal.

        Windows: Mở Anaconda Prompt từ Start Menu.

    Tạo môi trường mới:

        Gõ lệnh sau để tạo một môi trường tên là chatbot_env với Python 3.11.

        conda create --name chatbot_env python=3.11

        Khi được hỏi, gõ y và nhấn Enter.

    Kích hoạt môi trường:

        Để bắt đầu làm việc trong môi trường vừa tạo, hãy kích hoạt nó:

        conda activate chatbot_env

        Bạn sẽ thấy (chatbot_env) xuất hiện ở đầu dòng lệnh.

    Cài đặt thư viện:

        Dùng lệnh cd để di chuyển vào thư mục dự án của bạn.

        Chạy lệnh sau để cài đặt tất cả các gói cần thiết:

        pip install -r requirements.txt

**Bước 5: Chạy Hệ thống**

Bạn sẽ cần hai cửa sổ Terminal riêng biệt, cả hai đều đã được kích hoạt môi trường chatbot_env.

    Chạy Backend (API Server):

        Trong cửa sổ Terminal thứ nhất:
        Kích hoạt môi trường (nếu chưa làm)

        conda activate chatbot_env
        Chạy server

        uvicorn backend_app:app --reload

        Server sẽ khởi động và bạn sẽ thấy thông báo nó đang chạy tại https://www.google.com/search?q=http://127.0.0.1:8000. Hãy giữ cửa sổ này mở.

    Chạy Frontend (Giao diện Chat):

        Trong cửa sổ Terminal thứ hai:
        Kích hoạt môi trường (nếu chưa làm)

        conda activate chatbot_env
        Chạy ứng dụng giao diện

        streamlit run frontend_app.py

**Bước 6: Sử dụng Chatbot**

    Sau khi chạy lệnh streamlit, một tab trình duyệt mới sẽ tự động mở ra.

    Ở thanh bên trái, dán Hugging Face API Token của bạn vào ô yêu cầu.

    Sau khi token được chấp nhận, bạn có thể bắt đầu trò chuyện.

**Bước 7: Dừng Hệ thống**

    Trong mỗi cửa sổ Terminal, nhấn tổ hợp phím Ctrl + C để dừng server.

    Để thoát khỏi môi trường Anaconda, gõ lệnh:

    conda deactivate
