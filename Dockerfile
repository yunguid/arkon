# Multi-stage Dockerfile for Arkon Financial Intelligence Platform

# Stage 1: Build frontend
FROM node:18-alpine as frontend-builder
WORKDIR /app/frontend
COPY frontend/package*.json ./
RUN npm ci --only=production
COPY frontend/ ./
RUN npm run build

# Stage 2: Build backend dependencies
FROM python:3.11-slim as backend-builder
WORKDIR /app
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libffi-dev \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt ./
RUN pip install --user --no-cache-dir -r requirements.txt

# Stage 3: ML Model preparation
FROM python:3.11-slim as ml-builder
WORKDIR /ml
COPY backend/ml_engine.py ./
COPY backend/requirements.txt ./
RUN pip install --no-cache-dir torch torchvision prophet scikit-learn

# Stage 4: Final production image
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    nginx \
    supervisor \
    tesseract-ocr \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# Create app user
RUN useradd -m -s /bin/bash arkon

# Copy Python dependencies from builder
COPY --from=backend-builder /root/.local /home/arkon/.local

# Copy application files
WORKDIR /app
COPY --chown=arkon:arkon backend/ ./backend/
COPY --chown=arkon:arkon --from=frontend-builder /app/frontend/build ./frontend/build

# Copy ML models
COPY --from=ml-builder /ml ./ml

# Copy configuration files
COPY nginx.conf /etc/nginx/nginx.conf
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# Create necessary directories
RUN mkdir -p /app/logs /app/models /app/uploads /app/cache \
    && chown -R arkon:arkon /app

# Environment variables
ENV PYTHONPATH=/app/backend:/home/arkon/.local/lib/python3.11/site-packages
ENV PATH=/home/arkon/.local/bin:$PATH
ENV WORKERS=4
ENV PORT=8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose ports
EXPOSE 80 8000 8080

# Switch to app user
USER arkon

# Start services
CMD ["/usr/bin/supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"] 