import json
from config import config


def build_list_case_request():
    url = f"{config.api_url}/case"

    request_obj = {
        "method": "GET",
        "url": url,
    }

    return request_obj


def build_new_case_request(
    user_id="62ac47cf4f01fd1051d23912",
    api_key="62a19913a18e3218bbddeff2:FHH29498iifiedfsdfJFHJ//sdfa",
):
    url = f"{config.api_url}/case"

    data = {
        "apiKey": api_key,
        "userId": user_id,
        "type": "DAMAGE_ASSESSMENT",
    }

    request_obj = {
        "method": "POST",
        "url": url,
        "data": json.dumps(data),
        "headers": {"Content-Type": "application/json"},
    }

    return request_obj


def build_get_case_request(case_id):
    url = f"{config.api_url}/case/{case_id}"

    request_obj = {
        "method": "GET",
        "url": url,
    }

    return request_obj


def build_delete_case_request(case_id):
    url = f"{config.api_url}/case/{case_id}"

    request_obj = {
        "method": "DELETE",
        "url": url,
    }

    return request_obj


def build_get_case_status(case_id):
    url = f"{config.api_url}/case/{case_id}/status"

    request_obj = {
        "method": "GET",
        "url": url,
    }

    return request_obj
