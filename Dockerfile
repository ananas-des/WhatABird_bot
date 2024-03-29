FROM python:3.9.13
WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /app/requirements.txt && \
    pip install torch torchvision torchaudio -f https://download.pytorch.org/whl/cu111/torch_stable.html
COPY /source/bird_bot.py /app/bird_bot.py
COPY /source/model /app/model
ENTRYPOINT ["python", "/app/bird_bot.py"]
