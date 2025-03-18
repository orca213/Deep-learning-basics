FROM ubuntu:20.04

WORKDIR /workspace

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip

COPY requirements.txt /workspace/
RUN pip3 install --no-cache-dir --break-system-packages -r requirements.txt