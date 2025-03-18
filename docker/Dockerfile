FROM ubuntu:20.04

WORKDIR /workspace

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip

COPY requirements.txt /workspace/
RUN pip install --no-cache-dir -r requirements.txt