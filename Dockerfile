FROM ultralytics/yolov5:latest-cpu

WORKDIR /app

RUN apt-get install python-is-python3 nano \
  -y --no-install-recommends && \
  rm -rf /var/lib/apt/lists/*

COPY ./app requirements.txt /app/

RUN pip install -r requirements.txt

CMD ["/bin/bash", "/app/entrypoint.sh"]
