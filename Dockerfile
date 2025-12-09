FROM python:3.10

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .

# Install runtime dependencies only
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Run the evaluator by default. Additional CLI args can be appended to `docker run`.
ENTRYPOINT ["python", "EvaluateFile.py"]
CMD []