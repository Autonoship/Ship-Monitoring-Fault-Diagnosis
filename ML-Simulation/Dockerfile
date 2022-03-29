FROM python:3.9

RUN mkdir my-model
ENV MODEL_DIR=./my-model
ENV MODEL_FILE_LDA=clf_lda.joblib
ENV MODEL_FILE_NN=clf_nn.joblib

COPY requirements.txt ./requirements.txt
RUN pip install -r requirements.txt 

COPY train_ASC.csv ./train_ASC.csv
COPY test_ASC.csv ./test_ASC.csv

COPY backend.py ./backend.py
COPY frontend.py ./frontend.py

RUN python3 backend.py