import os
import json

from utils import get_image_filepaths, get_filename_from_filepath, image_to_data_url
from config import config

IMAGE_DIR = os.path.join(config.project_root, "images/")


def build_add_vehicle_request(
    case_id,
    user_id,
    api_key,
):
    url = f"{config.api_url}/case/{case_id}/add-vehicle/v2"

    path_list = get_image_filepaths(IMAGE_DIR)

    file_list = [
        {
            "fileName": get_filename_from_filepath(filepath),
            "base64Str": image_to_data_url(filepath),
        }
        for filepath in path_list
    ]

    data = {"apiKey": api_key, "userId": user_id, "fileList": file_list}

    request_obj = {
        "method": "POST",
        "url": url,
        "data": json.dumps(data),
        "headers": {"Content-Type": "application/json"},
    }

    return request_obj
