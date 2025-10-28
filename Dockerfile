# Dockerfile
FROM python:3.10-slim

WORKDIR /app

# copy only requirements first for layer caching
RUN apt-get update && apt-get install -y python3-pip build-essential
RUN pip3 install --upgrade pip
RUN pip3 install cython==0.29.36
RUN pip3 install --no-cache-dir -r requirements.txt

# copy project
COPY . .

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]


