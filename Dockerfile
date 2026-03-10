FROM python:3.11-slim

WORKDIR /app

# Dependency for importing LightGBM, which is stripped out in python slim
RUN apt-get update && apt-get install -y libgomp1 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY .streamlit ./.streamlit
COPY data ./data
COPY model ./model
COPY pages ./pages
COPY utils ./utils
COPY app.py ./

EXPOSE 8501

CMD ["python", "-m", "streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]