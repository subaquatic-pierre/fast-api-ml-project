# pull official base image
# FROM python:3.9.5-slim-buster
FROM nvidia/cuda:11.0.3-base-ubuntu20.04

# RUN apt-get install -y --no-install-recommends wget
# RUN apt-key del 7fa2af80
# RUN wget developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb
# RUN dpkg -i cuda-keyring_1.0-1_all.deb
# RUN apt-get update



# set work directory
WORKDIR /app

EXPOSE 4000
ENV DEBIAN_FRONTEND=noninteractive

# set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# install system dependencies
RUN apt-get update && apt-get install -y netcat
RUN apt install -y build-essential libssl-dev python-dev

# install dependencies
RUN apt update
RUN apt install -y python3-venv python3-pip
RUN python3 -m pip install --upgrade pip
COPY ./requirements.txt /app/requirements.txt
RUN python3 -m pip install wheel
# RUN apt install -y python3-setuptools 
RUN pip install -r requirements.txt

RUN apt install -y git
RUN python3 -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
RUN python3 -m pip install python-dotenv
RUN apt-get install ffmpeg libsm6 libxext6  -y

COPY . /app/

# CMD [ "python -m gunicorn -b 0.0.0.0:5000 -w 4 main:app" ]
