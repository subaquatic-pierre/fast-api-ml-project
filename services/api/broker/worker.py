from celery import Celery
from broker.config import CeleryConfig

celery = Celery(__name__)
celery.config_from_object(CeleryConfig())
