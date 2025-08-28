FROM python:3.10-slim
WORKDIR /app

# for "git+https://..." in requirements
RUN apt-get update && apt-get install -y --no-install-recommends git \
  && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY shap_e_cli.py .

# default output volume
VOLUME ["/out"]

# make CLI the entrypoint; users can append flags
ENTRYPOINT ["python", "/app/shap_e_cli.py"]
