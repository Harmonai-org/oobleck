FROM ubuntu:latest
RUN apt-get update && apt-get install -y python3.10 python3-pip python3.10-venv
WORKDIR /oobleck
COPY requirements.txt .
RUN pip install -r requirements.txt
RUN pip install build
COPY . .
CMD pytest && OOBLECK_VERSION=0.0.1 python3 -m build