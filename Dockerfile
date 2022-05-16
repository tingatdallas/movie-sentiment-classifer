FROM python:3.8

WORKDIR /app-home

COPY . /requirements.txt

RUN pip install --trusted-host pypi.python.org -r requirements.txt
Run pip install gunicorn

COPY . .

ENTRYPOINT ["python", "app.py"]