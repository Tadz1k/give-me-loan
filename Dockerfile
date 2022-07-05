FROM python:3.8-slim-buster

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends apt-utils
RUN apt-get -y install curl
RUN apt-get install libgomp1

RUN pip install pycaret
RUN pip install numpy
RUN pip install pandas
RUN pip install plotly
RUN pip install mlflow

COPY /app .
copy /app/train.py train.py


ENTRYPOINT ["python", "main.py"]