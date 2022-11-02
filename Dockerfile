FROM python:3.10.5
# Or any preferred Python version. (main, main -> temp, temp)
ADD temp.py .
RUN pip install scikit-learn numpy pandas
CMD ["python3", "temp.py"]
