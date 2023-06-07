from typing import List
import os
import matplotlib.image as mpimg

from helpers.images import save_cv2_image
from helpers.utils import get_case_dir
from helpers.s3 import download_file

from config.logger import logger
from config.config import settings


def save_top_view_images(case_id, final_top_view1, final_top_view2, both_car):
    save_cv2_image(case_id, final_top_view1, "final_top_view1.jpg")
    save_cv2_image(case_id, final_top_view2, "final_top_view2.jpg")
    save_cv2_image(case_id, both_car, "both_car.jpg")


def build_marker_list(case_id, model_dict: List, car_number) -> List:
    case_dir = get_case_dir(case_id)
    car_images = [car for car in model_dict if int(car["car_no"]) == car_number]
    remote_marker_iamge_paths: List[str] = [
        image["marker_path"] for image in car_images
    ]

    local_marker_image_paths = []

    for image_url in remote_marker_iamge_paths:
        filename = image_url.split("/")[-1]
        file_key = f"{case_id}/{filename}"
        local_path = os.path.join(case_dir, filename)
        download_file(file_key, local_path)
        local_marker_image_paths.append(local_path)

    return local_marker_image_paths


def process_top_view(case_id, model, model_dict):
    logger.info("Machine learning started for top view")

    BUCKET_URL = settings.bucket_url
    case_dir = get_case_dir(case_id)
    damage_model = model(case_dir)

    marker_list_1 = build_marker_list(case_id, model_dict, car_number=1)
    marker_list_2 = build_marker_list(case_id, model_dict, car_number=2)

    tp1 = mpimg.imread(marker_list_1[0])
    tp2 = mpimg.imread(marker_list_1[1])
    tp3 = mpimg.imread(marker_list_1[2])
    tp4 = mpimg.imread(marker_list_1[3])

    tp5 = mpimg.imread(marker_list_2[0])
    tp6 = mpimg.imread(marker_list_2[1])
    tp7 = mpimg.imread(marker_list_2[2])
    tp8 = mpimg.imread(marker_list_2[3])

    final_top_view1, final_top_view2, both_car = damage_model.composite_two_car(
        tp1, tp2, tp3, tp4, tp5, tp6, tp7, tp8
    )

    final_top_view1_filename = "final_top_view1.jpg"
    final_top_view2_filename = "final_top_view2.jpg"
    both_case_filename = "accident_diagram.jpg"

    final_top_view1.save(os.path.join(case_dir, final_top_view1_filename))
    final_top_view2.save(os.path.join(case_dir, final_top_view2_filename))
    both_car.save(os.path.join(case_dir, both_case_filename))

    # Generate new image urls
    final_top_view1_url = f"{BUCKET_URL}/{case_id}/{final_top_view1_filename}"
    final_top_view2_url = f"{BUCKET_URL}/{case_id}/{final_top_view2_filename}"
    accident_diagram = f"{BUCKET_URL}/{case_id}/{both_case_filename}"

    data = {
        "final_top_view1": final_top_view1_url,
        "final_top_view2": final_top_view2_url,
        "accident_diagram": accident_diagram,
    }

    logger.info("Machine learning completed for top view")

    return data
