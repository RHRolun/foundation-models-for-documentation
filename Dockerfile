FROM python:3.10.4

WORKDIR /app

COPY app/requirements.txt ./requirements.txt

RUN pip install -r requirements.txt

RUN pip list > piplist.txt

# override default port (from 8501 to 8080)
ENV STREAMLIT_SERVER_PORT=8080
# these variables might be needed for the right info (logging) to show up in the log
ENV PYTHONUNBUFFERED=1
ENV PYTHONIOENCODING=UTF-8
ENV TRANSFORMERS_CACHE=/app
ENV HUGGINGFACE_HUB_CACHE=/app
ENV SENTENCE_TRANSFORMERS_HOME=/app

EXPOSE 8080

COPY ./app ./app
COPY ./data ./data

WORKDIR /app/app

RUN python -c "from langchain.embeddings import HuggingFaceEmbeddings; embeddings = HuggingFaceEmbeddings()"

ENTRYPOINT ["streamlit", "run"]

CMD ["frontend_app.py"]
