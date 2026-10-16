FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8000 8501
# API:       docker run -p 8000:8000 <img> uvicorn api.main:app --host 0.0.0.0 --port 8000
# Dashboard: docker run -p 8501:8501 <img> streamlit run app/dashboard.py --server.port 8501 --server.address 0.0.0.0
CMD ["streamlit", "run", "app/dashboard.py", "--server.port", "8501", "--server.address", "0.0.0.0"]
