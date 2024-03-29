# config.py

import os

class Config:
    MONGO_DB_URI = os.environ.get('MONGO_DB_URI')
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'some-secret-key'
    MAX_MODELS = 5  # Maximum number of concurrent BOPE models
