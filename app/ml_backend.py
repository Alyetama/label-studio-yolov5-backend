#!/usr/bin/env python
# coding: utf-8

import requests
from label_studio_ml.model import LabelStudioMLBase
from loguru import logger
from requests.exceptions import HTTPError


class MyModel(LabelStudioMLBase):

    def __init__(self, **kwargs):
        super(MyModel, self).__init__(**kwargs)
        self.api_endpoint = 'http://0.0.0.0:8000/predict'

    def predict(self, tasks, **kwargs):
        predictions = []

        for task in tasks:
            data = {'task': task, 'project': task['project']}
            try:
                r = requests.post(self.api_endpoint, json=data)
            except HTTPError as e:
                logger.error(e)
                continue
            if r.status_code != 200:
                logger.error(r.text)
                continue
            pred = r.json()
            predictions.append(pred)

        return predictions
