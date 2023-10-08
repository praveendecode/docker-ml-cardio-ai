FROM python:3.10
WORKDIR /app
COPY heart.py .
COPY heart_attack.csv .
RUN pip install scikit-learn \
pandas 
CMD python heart.py
