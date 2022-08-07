#!/bin/bash

label-studio-ml init label-studio-yolov5-backend --script ml_backend.py

nohup label-studio-ml start ./label-studio-yolov5-backend &

python3 api.py
