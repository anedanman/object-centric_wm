FROM python:3.8
WORKDIR /code/
RUN apt-get update
RUN apt-get install zip build-essentials ffmpeg libsm6 libxext6 wget
COPY . .
RUN pip install -r requrements.txt
ENTRYPOINT ["tail", "-f", "/dev/null"]