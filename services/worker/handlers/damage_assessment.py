from concurrent.futures import ThreadPoolExecutor
import time

from config.config import settings
from helpers.images import save_cv2_images
from helpers.reports import build_inner_model_dict
from config.logger import logger


def process_image(case_id, damage_model, image_path, image_id, vehicle_number):
    bucket_url = settings.bucket_url

    # Main Machine learning prediction per image
    start = time.time()
    (
        img,
        dmgMaskImg,
        outClass,
        panel_label,
        TPview,
        final_dict,
        damage_type_of_car,
    ) = damage_model.process_image(image_path)
    end = time.time()

    # Save images from model prediction
    filename = save_cv2_images(case_id, image_path, img, dmgMaskImg, TPview)

    model_process_duration = end - start
    bucket_prefix = f"{bucket_url}/{case_id}"

    inner_model_dict = build_inner_model_dict(
        final_dict,
        damage_type_of_car,
        model_process_duration,
        bucket_prefix,
        filename,
        panel_label,
        image_id,
        vehicle_number,
    )

    return inner_model_dict


def process_damage_assessment(case_id, damage_model, image_path_list, vehicle_number=1):
    """
    Main entry point to start prediction on list of images. List of images are passed to method as List[str]

    Params:
    case_id: str
    model: Machine Learning model
    image_path_list: List[str]
    case_dir: str
    bucket_url: str

    Returns:
    report: dict
    """
    logger.info("Machine learning started")

    car_report = []
    results = []

    with ThreadPoolExecutor(max_workers=8) as pool:
        for img_id, image_path in enumerate(image_path_list):

            process_image_args = {
                "case_id": case_id,
                "damage_model": damage_model,
                "image_path": image_path,
                "image_id": img_id,
                "vehicle_number": vehicle_number,
            }

            result = pool.submit(process_image, **process_image_args)
            results.append(result)

        for result in results:
            inner_model_dict = result.result()
            car_report.append(inner_model_dict)

    if len(car_report) > 0:
        logger.info("Machine learning complete")
        return car_report
    else:
        raise Exception("Machine learning model failed")
