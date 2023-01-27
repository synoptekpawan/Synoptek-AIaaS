# FROM python:3.9
# WORKDIR /retailAnalytics
# COPY requirements.txt ./requirements.txt
# RUN pip3 install -r requirements.txt
# EXPOSE 8501
# COPY . /retailAnalytics/
# ENTRYPOINT ["streamlit", "run"]
# CMD ["Home.py", "--server.headless=true"]

FROM python:3.9

WORKDIR /retailAnalytics

COPY requirements.txt ./requirements.txt

RUN pip3 install -r requirements.txt

EXPOSE 8501

COPY . .

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "Home.py","--server.port=8501", "--server.address=0.0.0.0",  "--server.headless=true"]


