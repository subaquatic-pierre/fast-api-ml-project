from concurrent.futures import as_completed
from pprint import pprint
import os
import base64
from time import time
from typing import List
from case import build_get_case_request
from requests_futures.sessions import FuturesSession
import datetime


from log import logger


session = FuturesSession(max_workers=10)


def image_to_data_url(filepath):
    ext = filepath.split(".")[-1]
    prefix = f"data:image/{ext};base64,"
    with open(filepath, "rb") as f:
        img = f.read()
    return prefix + base64.b64encode(img).decode("utf-8")


def get_image_filepaths(image_dir) -> List[str]:
    filepath_list = []
    for file in os.scandir(image_dir):
        filepath_list.append(file.path)

    return filepath_list


def get_filename_from_filepath(filepath):
    _, tail = os.path.split(filepath)
    return tail


def logger_dec(func):
    def wrap_func(*args, **kwargs):
        result = func(*args, **kwargs)
        msg = f"Function {func.__name__!r} executed"
        logger.info(msg)
        print(msg)
        return result

    return wrap_func


def timer(func):
    def wrap_func(*args, **kwargs):
        t1 = time()
        result = func(*args, **kwargs)
        t2 = time()
        msg = f"Function {func.__name__!r} executed in {(t2-t1):.2f}s"
        logger.info(msg)
        print(msg)
        return result

    return wrap_func


def get_case_ids(case_list: List[dict]):
    case_ids = []
    for case in case_list:
        case_ids.append(case["id"])

    return case_ids


def handle_request_futures(futures):
    responses = []
    response_data = []
    for future in as_completed(futures):
        resp = future.result()

        # Append responses to response list
        responses.append(resp)

        # Get data from response
        data = resp.json()["data"]

        # for response in responses:
        #     pprint(
        #         {
        #             "url": response.request.url,
        #             "data": data,
        #         }
        #     )

        response_data.append(data)

    return response_data


def check_all_case_status(case_ids: List[str], key: str, value):
    all_case_status = False
    futures = [
        session.request(**build_get_case_request(case_ids[i]))
        for i in range(len(case_ids))
    ]
    data = handle_request_futures(futures)

    status = [case[key] for case in data]
    all_case_status = all([item == value for item in status])

    return all_case_status


def start_log():
    logger.info("\n--- NEW RUN ---")
    logger.info(f"DATE: {datetime.datetime.now()}")
    logger.info("--- --- ---")


def end_log():
    logger.info("--- --- ---")
    logger.info(f"DATE: {datetime.datetime.now()}")
    logger.info("--- END ---")
