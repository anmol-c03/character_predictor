FROM pytorch/pytorch

RUN mkdir -p /home/wavenet

RUN apt update && \
    apt install -y nano

COPY . /home/wavenet

WORKDIR /home/wavenet


CMD [ "python3" ,"wavenet.py" ]


