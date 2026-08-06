# ==========================================================
# Base Image
# ==========================================================
FROM python:3.11-slim

# ==========================================================
# Python Environment
# ==========================================================
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV NLTK_DATA=/usr/local/share/nltk_data

# ==========================================================
# Working Directory
# ==========================================================
WORKDIR /app

# ==========================================================
# Install Runtime Dependencies
# ==========================================================
COPY requirements ./requirements

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements/inference.txt

# ==========================================================
# Download NLTK Resources
# ==========================================================
RUN mkdir -p ${NLTK_DATA} && \
    python -m nltk.downloader -d ${NLTK_DATA} stopwords wordnet && \
    chmod -R 755 ${NLTK_DATA}

# ==========================================================
# Copy Application
# ==========================================================
COPY . .

# ==========================================================
# Create Non-Root User
# ==========================================================
RUN addgroup --system appgroup && \
    adduser --system --ingroup appgroup appuser && \
    chown -R appuser:appgroup /app

USER appuser

# ==========================================================
# Application Port
# ==========================================================
EXPOSE 5000

# ==========================================================
# Docker Healthcheck
# ==========================================================
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:5000/health', timeout=3)" || exit 1

# ==========================================================
# Production Server
# ==========================================================
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "2", "--threads", "2", "--timeout", "60", "flask_app.app:app"]