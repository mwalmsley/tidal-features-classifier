FROM python:3.6

COPY . ~/tidalclassifier

RUN pip install -r ~/repo_root/requirements.txt  # does not include tensorflow, keras, seaborn
RUN pip install tensorflow==1.11.0
RUN pip install keras==2.2.4
RUN pip install photutils --no-deps

# do this last to exploit caching
WORKDIR ~/tidalclassifier
RUN pip install -e tidalclassifier
