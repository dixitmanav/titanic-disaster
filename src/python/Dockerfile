# Build from the repo root:
#   docker build -f src/python/Dockerfile -t titanic-python .
FROM python:3.12-slim

WORKDIR /app

# Install packages first so this layer is cached between code changes
COPY src/python/requirements.txt src/python/requirements.txt
RUN pip install --no-cache-dir -r src/python/requirements.txt

# Copy the code and the (locally downloaded) data
COPY src/python/ src/python/
COPY src/data/ src/data/

CMD ["python", "src/python/titanic_model.py"]
