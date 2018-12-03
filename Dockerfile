FROM python:3.6

WORKDIR /app

COPY requirements.txt .

RUN pip install -r requirements.txt

ENV DEBUG=False
ENV LISTEN_PORT=8080
ENV WORKERS=2
ENV WORKERCLASS=sync
ENV LOG_LEVEL=DEBUG

COPY . .

ENTRYPOINT [ "./bin/entrypoint.sh" ]
