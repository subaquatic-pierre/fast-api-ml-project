import os

broker_url = os.environ.get("CELERY_BROKER_URL", "redis://localhost:6379")
backend_url = ("CELERY_RESULT_BACKEND", "redis://localhost:6379")


class CeleryConfig:
    CELERY_BROKER_URL = broker_url
    CELERY_RESULT_BACKEND = backend_url
    CELERY_IMPORTS = [
        "handlers.assessments",
        "handlers.case",
        "handlers.reports",
    ]
    CELERY_RESULT_SERIALIZER = "pickle"
    CELERY_TASK_SERIALIZER = "pickle"
    CELERY_ACCEPT_CONTENT = ["pickle", "application/text"]
