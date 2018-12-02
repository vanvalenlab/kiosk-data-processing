FROM python:3.6

WORKDIR /app

ENV DEBUG=False
ENV LISTEN_PORT=8080

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY . .

ENTRYPOINT [ "./bin/entrypoint.sh" ]
