FROM python:3.12-slim-bookworm

WORKDIR /usr/src/app

COPY . .

RUN pip install --no-cache-dir --upgrade pip \
  && pip install --no-cache-dir -r requirements.txt

CMD ["fastapi", "run", "main.py", "--proxy-headers", "--port", "80"]

