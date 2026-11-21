# 运行镜像。核心逻辑与 HTTP 服务只用标准库，docx / pdf 解析依赖见 requirements.txt。
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt ./
RUN pip install -r requirements.txt

COPY . .

ENV AGENT_DB=/app/state/tasks.db
RUN mkdir -p /app/state && useradd --create-home --shell /usr/sbin/nologin app \
    && chown -R app:app /app/state

RUN python -m unittest discover -s tests

USER app
VOLUME ["/app/state"]

EXPOSE 8081

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import sys,urllib.request; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8081/healthz', timeout=2).status == 200 else 1)"

# 默认启动 HTTP 服务；只想跑一次 CLI 时：
#   docker run --rm <image> python -m agent.cli --doc d3
ENTRYPOINT ["python"]
CMD ["-m", "agent.api"]
