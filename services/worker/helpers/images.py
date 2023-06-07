from pathlib import Path
from typing import List
import base64
import os
import cv2

from helpers.utils import get_case_dir


def save_cv2_image(case_id, image, filename):
    case_dir = get_case_dir(case_id)
    file_path = str(Path(case_dir, filename))
    cv2.imwrite(file_path, cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


def save_cv2_images(case_id, image_path, img, dmgMaskImg, TPview):
    # split the file name and path
    _, tail = os.path.split(image_path)
    filename = tail.split(".")[0]

    # write final image as output
    output_filename = f"{filename}_annotated.jpg"
    save_cv2_image(case_id, img, output_filename)

    # write damage image as output
    dmg_filename = f"{filename}_damage_annotated.jpg"
    save_cv2_image(case_id, dmgMaskImg, dmg_filename)

    # write marker image (top view)
    marker_img_filename = f"{filename}_marker.jpg"
    save_cv2_image(case_id, TPview, marker_img_filename)

    return filename


def save_images(case_id: str, file_list, image_prefix=1) -> List[str]:
    case_dir = get_case_dir(case_id)
    # Init new file path list to be returned
    file_path_list = []

    for file in file_list:
        # Generate new path for file
        filename = f'{image_prefix}_{file["fileName"]}'

        output_file = Path(case_dir, filename)

        base64_string = file.get("base64Str").split(",")[1]
        base64_image = base64.decodebytes(base64_string.encode())
        output_file.write_bytes(base64_image)

        # Add file path to path list
        file_path_list.append(str(output_file))

    return file_path_list


def remove_marker_images(case_id):
    case_dir = get_case_dir(case_id)
    for file in os.listdir(case_dir):
        pass
