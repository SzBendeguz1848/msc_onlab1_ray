FROM python:3.11-slim
WORKDIR /app
COPY pyproject.toml ./
COPY src ./src
RUN pip install --no-cache-dir -e ".[serve]"
ENV PYTHONUNBUFFERED=1
# router.py starts Ray + serve.run; binds Serve HTTP on 8000
CMD ["python", "src/router.py"]
