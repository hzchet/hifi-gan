FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-devel

RUN apt update && apt install -y sudo
RUN apt-get install wget -y
RUN apt install unzip -y

# installing required packages
COPY requirements.txt ./
RUN pip install -r requirements.txt

ENV WANDB_API_KEY=8e9008b623a334edf472f175d059c25c9aa66207
