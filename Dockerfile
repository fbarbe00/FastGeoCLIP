FROM python:3.11-slim

WORKDIR /app

# Set environment variables to reduce warnings and image size
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    TF_CPP_MIN_LOG_LEVEL=3 \
    TOKENIZERS_PARALLELISM=false \
    PIP_NO_CACHE_DIR=1

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libopenblas0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies with aggressive cleanup in same layer
COPY requirements.txt .
RUN pip install --no-cache-dir -q --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt && \
    # Cleanup of unnecessary files to reduce image size
    find /usr/local/lib/python3.11/site-packages -type d \( -name "tests" -o -name "test" -o -name "__pycache__" \) -exec rm -rf {} + 2>/dev/null || true && \
    find /usr/local/lib/python3.11/site-packages -type f \( -name "*.pyc" -o -name "*.pyo" \) -delete 2>/dev/null && \
    # Remove documentation and examples
    find /usr/local/lib/python3.11/site-packages -type d \( -name "docs" -o -name "doc" -o -name "examples" -o -name "example" \) -exec rm -rf {} + 2>/dev/null || true && \
    rm -rf /root/.cache 2>/dev/null || true

# Copy application code, package, and bundled weights/GPS gallery.
# CLIP vision tower (clip/) and GeoPackage (data/) are mounted at runtime.
COPY app.py .
COPY fastgeoclip/ ./fastgeoclip/

# Expose port
EXPOSE 8000

# Health check (simplified socket check instead of HTTP request)
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import socket; socket.create_connection(('localhost', 8000), timeout=5)" || exit 1

# Run application with warning suppression
CMD ["python", "-W", "ignore::FutureWarning", "app.py"]
