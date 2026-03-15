# Base: Python + Node.js (needed for Claude Code CLI)
FROM python:3.12-slim

# --- System dependencies ---
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    git \
    ca-certificates \
    gnupg \
    && rm -rf /var/lib/apt/lists/*

# --- Install Node.js 22 (LTS) ---
RUN curl -fsSL https://deb.nodesource.com/setup_22.x | bash - \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/*

# --- Install Claude Code CLI globally ---
RUN npm install -g @anthropic-ai/claude-code

# --- Python dependencies ---
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# --- Copy project files ---
COPY . .

# Create output data directory
RUN mkdir -p data

# Default: drop into an interactive shell so you can trigger scripts or claude manually
CMD ["/bin/bash"]
