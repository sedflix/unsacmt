FROM tensorflow/tensorflow:2.0.0-gpu-py3-jupyter

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
RUN PYTHONIOENCODING=utf-8 python -m laserembeddings download-models

CMD ["bash"]
