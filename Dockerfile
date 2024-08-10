# # base image
# FROM python:3.10.12-slim

# # work folder
# WORKDIR /app

# # copy file in container
# COPY requirements.txt .

# # # setup latest pip
# # RUN pip install --upgrade pip

# #setup
# RUN pip install --no-cache-dir -r requirements.txt

# # copy project in container
# COPY . .

# # run script to download file pre-trained
# RUN python download_pre_trained.py

# # CLI run fastapi
# CMD ["python", "main.py"]

FROM python:3.12

# Cài đặt các thư viện hệ thống cần thiết
RUN apt-get update && apt-get install -y libgl1-mesa-glx

# Sao chép mã nguồn vào container
WORKDIR /app
COPY . /app

# Cài đặt các phụ thuộc
RUN pip install -r requirements.txt

# Chạy ứng dụng Streamlit
CMD ["streamlit", "run", "app.py"]
