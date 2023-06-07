import base64
from typing import List
import pdfkit
import os
import json
import requests

from config.config import settings
from helpers.utils import get_case_dir
from config.logger import logger


def build_inner_model_dict(
    final_dict,
    damage_type_of_car,
    model_process_duration,
    bucket_prefix,
    filename,
    panel_label,
    image_id,
    vehicle_number,
):
    original_path = f"{bucket_prefix}/{filename}.jpg"
    final_path = f"{bucket_prefix}/{filename}_annotated.jpg"
    damage_path = f"{bucket_prefix}/{filename}_damage_annotated.jpg"
    marker_img = f"{bucket_prefix}/{filename}_marker.jpg"

    label_text = []
    tx = 0
    for key, vals in final_dict.items():

        value = []
        scores = []
        for v in vals:
            value.append(list(v.keys())[0])
            scores.append(list(v.values())[0])

        lt = {}
        lt["part"] = key.title()
        valstr = str(set(value))
        valstr = valstr.replace("{", "")
        valstr = valstr.replace("}", "")
        valstr = valstr.replace("'", "")
        lt["issue"] = (
            "No Damage"
            if value == []
            else valstr.title().replace("bumper fix", "bumper crack")
        )

        repair_status = ""
        price = "0.00"
        labcharge = "0.00"
        if lt["issue"] != "No Damage":
            if any(
                key in value
                for key in [
                    "bumper replace",
                    "bumper fix",
                    "bumper crack",
                    "crack",
                    "dent major",
                    "replace",
                    "front windshield",
                    "rear windshield",
                    "right tail lamp",
                    "left tail lamp",
                    "left indicator",
                    "right indicator",
                    "right headlamp",
                    "left headlamp",
                ]
            ):
                repair_status = "Replace"
                price = "800.00"
                labcharge = "200.00"
                tx = tx + 1
            else:
                repair_status = "Repair"
                price = "500.00"
                labcharge = "200.00"
                tx = tx + 1

        lt["repair_status"] = repair_status
        lt["price"] = price
        lt["labcharge"] = labcharge
        label_text.append(lt)

    inner_model_dict = {}
    inner_model_dict["img_id"] = str(image_id)
    inner_model_dict["car_no"] = str(vehicle_number)
    inner_model_dict["original_path"] = original_path
    inner_model_dict["image_path"] = final_path
    inner_model_dict["damage_path"] = damage_path
    inner_model_dict["marker_path"] = marker_img
    inner_model_dict["panel_text"] = list(set(panel_label))
    inner_model_dict["label_text"] = label_text
    inner_model_dict["process_timing"] = "%.2f" % model_process_duration
    inner_model_dict["damage_type_of_car"] = (
        "No Damage"
        if tx == 0
        else "Minor"
        if damage_type_of_car == []
        else damage_type_of_car.title()
    )

    return inner_model_dict


def build_pdf_report_branding():
    PROJECT_ROOT = settings.project_root
    with open(f"{PROJECT_ROOT}/static/img/DezzexLogo.png", "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
        logo = "data:image/png;base64," + encoded_string.decode("utf-8")

    with open(f"{PROJECT_ROOT}/static/img/Group12.png", "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
        bgd = "data:image/png;base64," + encoded_string.decode("utf-8")

    return logo, bgd


def build_pdf_report_rows(model_dict) -> List:
    rows = []
    for md in model_dict:
        if "original_path" in md:
            col = {}
            res = requests.get(md["original_path"])
            image_content = res.content
            encoded_string = base64.b64encode(image_content)
            col["original"] = "data:image/png;base64," + encoded_string.decode("utf-8")

            res = requests.get(md["damage_path"])
            image_content = res.content
            encoded_string = base64.b64encode(image_content)
            col["damage_annot"] = "data:image/png;base64," + encoded_string.decode(
                "utf-8"
            )

            col["damage_label"] = md["label_text"]
            col["damage_status"] = md["damage_type_of_car"]
            rows.append(col)

    return rows


def build_pdf_report_diagram(model_dict) -> dict:
    diagram = {}

    res = requests.get(model_dict[-1]["final_top_view1"])
    image_content = res.content
    encoded_string = base64.b64encode(image_content)
    diagram["final_top_view1"] = "data:image/png;base64," + encoded_string.decode(
        "utf-8"
    )

    res = requests.get(model_dict[-1]["final_top_view2"])
    image_content = res.content
    encoded_string = base64.b64encode(image_content)
    diagram["final_top_view2"] = "data:image/png;base64," + encoded_string.decode(
        "utf-8"
    )

    res = requests.get(model_dict[-1]["accident_diagram"])
    image_content = res.content
    encoded_string = base64.b64encode(image_content)
    diagram["accident_diagram"] = "data:image/png;base64," + encoded_string.decode(
        "utf-8"
    )

    return diagram


def save_pdf(file, path):
    options = {
        "page-size": "Letter",
        "footer-right": "[page] of [topage]",
    }
    pdfkit.from_string(file, path, options=options)


def save_json_report(case_id, model_dict):
    logger.info("Saving report")
    case_dir = get_case_dir(case_id)
    with open(os.path.join(case_dir, "report.json"), "w") as output:
        output.write(json.dumps(model_dict))
    logger.info("Report saved")
