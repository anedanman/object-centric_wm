FROM python:3.8
WORKDIR /code/
RUN apt-get update
RUN apt-get -y install zip build-essential ffmpeg libsm6 libxext6 wget
COPY . .
RUN pip install -r requirements.txt
ENTRYPOINT ["tail", "-f", "/dev/null"]