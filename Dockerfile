FROM python:3.8
WORKDIR /code
ENV PYTHONUNBUFFERED=1
RUN apt-get clean && apt-get update && apt-get install -y \
    ffmpeg \
    libsm6 \
    libxext6
COPY src/requirements.txt .
RUN pip install -r requirements.txt
COPY src .
ENTRYPOINT [ "python" ]
CMD [ "app.py" ]
