# Label Studio YOLOv5 Backend

## Getting started

1. Clone the repository

```bash
git clone https://github.com/Alyetama/label-studio-yolov5-backend.git
cd label-studio-yolov5-backend
```

2. Rename and edit `models_config_example.json`

```bash
mv models_config_example.json models_config.json
nano models_config.json  # or any other editor
```

3. Rename and edit `.env`

```bash
mv .env.example .env
nano .env  # or any other editor
```

## Running Locally

```bash
git clone https://github.com/ultralytics/yolov5.git

label-studio-ml init label-studio-yolov5-backend --script app/ml_backend.py
nohup label-studio-ml start ./label-studio-yolov5-backend &
```

## Running on Docker

### Build

You can skip this step if you want to use the pre-built docker image.

```bash
docker build . -t label-studio-yolov5-backend
```

### Run

- `8080` is the API port, and `9090` is the ML backend port.

- Run with `docker run`:

```bash
docker run -d \
  -v ${PWD}/weights:/app/weights \
  --env-file .env \
  -p 8000:8000 \
  -p 9090:9090 \
  alyetama/label-studio-yolov5-backend:latest
```

- Or with `docker-compose`:

```bash
docker-compose up -d
```
