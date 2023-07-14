# FROM python:3.9
# WORKDIR /retailAnalytics
# COPY requirements.txt ./requirements.txt
# RUN pip3 install -r requirements.txt
# EXPOSE 8501
# COPY . /retailAnalytics/
# ENTRYPOINT ["streamlit", "run"]
# CMD ["Home.py", "--server.headless=true"]

# Use a base image with Python pre-installed
FROM python:3.9

WORKDIR /reatilanalytics

COPY requirements.txt ./requirements.txt

RUN pip install -r requirements.txt

EXPOSE 8501

COPY . .

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["python","-m","streamlit", "run"]

CMD ["Home.py"]



