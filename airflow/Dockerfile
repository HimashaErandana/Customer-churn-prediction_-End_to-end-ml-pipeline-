FROM apache/airflow:3.3.2

USER root
COPY requirements.txt /requirements.txt
RUN pip install --no-cache-dir \
    apache-airflow==${AIRFLOW_VERSION} \
    apache-airflow-providers-standard \
    -r /requirements.txt
USER airflow
