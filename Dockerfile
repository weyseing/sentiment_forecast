# Python + Node.js
FROM python:3.12-slim

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    git \
    ca-certificates \
    gnupg \
    && rm -rf /var/lib/apt/lists/*

# Node.js 22
RUN curl -fsSL https://deb.nodesource.com/setup_22.x | bash - \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/*

# Terminal colors
ENV TERM=xterm-256color
ENV COLORTERM=truecolor
ENV CLICOLOR_FORCE=1

# Claude Code CLI
RUN npm install -g @anthropic-ai/claude-code

# Python dependencies
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Project files
COPY . .
RUN mkdir -p data

# Keep container alive
CMD ["tail", "-f", "/dev/null"]
