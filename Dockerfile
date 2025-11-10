FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app

# ✅ Render will use this port for health checks
EXPOSE 10000

# ✅ Run API on the correct port for Render
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "10000"]
