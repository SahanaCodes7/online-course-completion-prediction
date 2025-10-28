# Dockerfile
FROM python:3.10-slim

WORKDIR /app

# copy only requirements first for layer caching
RUN pip install --upgrade pip
RUN pip install cython==0.29.36
RUN apt-get update && apt-get install -y build-essential
RUN pip install --no-cache-dir -r requirements.txt

# copy project
COPY . .

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]


