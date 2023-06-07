# Accident AI Worker

## Dependencies

```sh
    sudo apt install -y \
    ffmpeg \
    libsm6 \
    libxext6 \
    libssl-dev \
    python-dev \
    build-essential
```

### Python requirements

```sh
    python -m pip install -r requirements.txt
```

### wkhtmltopdf

```sh
    wget https://github.com/wkhtmltopdf/packaging/releases/download/0.12.6-1/wkhtmltox_0.12.6-1.focal_amd64.deb
    sudo apt install ./wkhtmltox_0.12.6-1.focal_amd64.deb
```

reference: <https://computingforgeeks.com/install-wkhtmltopdf-on-ubuntu-debian-linux/>

### Detectron

```sh
    python3 -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
```

### Restart services

sudo systemctl restart nginx accident-ai-worker
