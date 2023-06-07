from requests_futures.sessions import FuturesSession
from pprint import pprint
from requests import request

from add_vehicle import build_add_vehicle_request

from utils import handle_request_futures, timer, logger_dec

from case import (
    build_get_case_status,
    build_new_case_request,
    build_list_case_request,
    build_delete_case_request,
    build_get_case_request,
)

USER_ID = "62ac47cf4f01fd1051d23912"
API_KEY = "62ac47cf4f01fd1051d23912:CytRyOg574YO-Wgi-mIqTQ"

session = FuturesSession(max_workers=10)


@timer
def make_list_cases_request():
    list_cases_request_obj = build_list_case_request()
    response = request(**list_cases_request_obj)

    data = response.json()["data"]

    return data


@logger_dec
def make_delete_cases_request(case_ids):
    futures = [
        session.request(**build_delete_case_request(case_ids[i]))
        for i in range(len(case_ids))
    ]

    data = handle_request_futures(futures)

    return data


@logger_dec
def make_new_case_request(num_cases):
    # Create new case request
    new_case_request_obj = build_new_case_request(USER_ID, API_KEY)

    futures = [session.request(**new_case_request_obj) for _ in range(num_cases)]

    data = handle_request_futures(futures)

    return data


@logger_dec
def make_add_vehicle_request(case_ids):
    # Create add vehicle request
    futures = [
        session.request(**build_add_vehicle_request(case_ids[i], USER_ID, API_KEY))
        for i in range(len(case_ids))
    ]

    data = handle_request_futures(futures)

    return data


def make_get_status_request(case_id):
    get_status_request_obj = build_get_case_status(case_id)
    response = request(**get_status_request_obj)

    data = response.json()["data"]
    status = data["status"]
    return status


def make_get_case_request(case_id):
    get_status_request_obj = build_get_case_request(case_id)
    response = request(**get_status_request_obj)

    data = response.json()["data"]
    return data
