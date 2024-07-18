# Use the Python 3.10 slim base image
FROM python:3.11-slim

# Set the working directory
WORKDIR /app

# Copy the requirements.txt file into the container
COPY requirements.txt .

# update pip from tsinghua mirror
# RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

# RUN pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/

# RUN pip config set global.index-url https://mirrors.cloud.tencent.com/pypi/simple

# Increase the stack size for new threads in the container
# RUN ulimit -s 2048

# # 设置最大线程数为 2048
# ENV PTHREAD_THREADS_MAX=2048

# # 禁用 pip 版本检查
# ENV PIP_DISABLE_PIP_VERSION_CHECK=1 
# # 启用详细日志记录 
# ENV PIP_VERBOSE=1  

# Upgrade pip
RUN pip install --upgrade pip

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

# Expose the port your FastAPI app will run on
EXPOSE 8000

# Start the FastAPI app using Uvicorn
CMD ["fastapi","run", "main.py", "--host", "0.0.0.0", "--port", "8000"]

# Start the Flask app
# CMD ["python", "app.py"]

# # Expose the port your Flask app will run on
# EXPOSE 5000

# # Start the Flask app with Gunicorn
# CMD ["gunicorn", "-b", ":5000", "your_flask_app:app"]