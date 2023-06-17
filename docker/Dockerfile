FROM python:3.9

LABEL maintainer="ycui982@aucklanduni.ac.nz"

RUN apt-get update \
    && apt autoremove -yqq --purge \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

ENV PYTHONPATH="${PYTHONPATH}:/ml_app"
WORKDIR "/ml_app"

COPY docker/requirements.txt .
COPY ../src .

RUN pip install --no-cache-dir --upgrade pip setuptools wheel \
    && pip install --no-cache-dir --progress-bar off -r requirements.txt \
    && rm -rf /var/lib/apt/lists/*

EXPOSE 80
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "80"]
