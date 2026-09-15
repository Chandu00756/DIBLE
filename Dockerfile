FROM python:3.12-slim
WORKDIR /app
COPY . .
RUN pip install --no-cache-dir -r requirements-server.txt
ENV DIBLE_DB_PATH=/var/lib/dible/dible.db
RUN useradd --system --create-home dible && mkdir -p /var/lib/dible && chown -R dible:dible /app /var/lib/dible
USER dible
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=3s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/health')"
CMD ["python", "-m", "dible_server.app"]
