from config.config import settings


def get_worker_url():
    return settings.worker_url


def parse_worker_res(res):
    return res.json()
