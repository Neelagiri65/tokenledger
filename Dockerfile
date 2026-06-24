FROM python:3.12-slim

WORKDIR /app
COPY pyproject.toml README.md ./
COPY retoken ./retoken
COPY examples ./examples

# Install with the `exact` extra so tiktoken is present for exact output-token verification.
RUN pip install --no-cache-dir ".[exact]"

# Data + reports live on a mounted volume so nothing leaves the box.
VOLUME ["/data"]
WORKDIR /data

ENTRYPOINT ["retoken"]
CMD ["--help"]
