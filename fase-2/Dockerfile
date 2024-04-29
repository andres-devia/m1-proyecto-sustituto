# use the image with the latest version of Python
FROM python:3.9

WORKDIR /app

COPY requirements.txt /app/requirements.txt
COPY . /app/

RUN pip install --no-cache-dir --upgrade -r requirements.txt

CMD ["/bin/bash"]
