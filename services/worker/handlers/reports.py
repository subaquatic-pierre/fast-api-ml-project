from flask import render_template
import jinja2
import datetime
import pdfkit

from config.config import settings
from helpers.reports import (
    build_pdf_report_branding,
    build_pdf_report_rows,
    build_pdf_report_branding,
    build_pdf_report_rows,
    build_pdf_report_diagram,
)


def render_jinja_html(file_name, data):
    template_loc = f"{settings.project_root}/templates/"

    return (
        jinja2.Environment(loader=jinja2.FileSystemLoader(template_loc))
        .get_template(file_name)
        .render(**data)
    )


def generate_damage_assessment_pdf_report(model_dict) -> bytes:
    options = {
        "page-size": "Letter",
        "footer-right": "[page] of [topage]",
    }

    data = {}

    # Download images to case dir

    logo, bgd = build_pdf_report_branding()
    rows = build_pdf_report_rows(model_dict)

    data["logo"] = logo
    data["bgd"] = bgd
    data["row"] = rows
    data["print_date"] = datetime.datetime.now().strftime("%d/%m/%Y, %H:%M:%S")

    render = render_jinja_html("report.html", {"datas": data})

    pdf_file = pdfkit.from_string(render, False, options=options)

    return pdf_file


def generate_top_view_pdf_report(model_dict) -> bytes:
    options = {
        "page-size": "Letter",
        "footer-right": "[page] of [topage]",
    }

    data = {}

    logo, bgd = build_pdf_report_branding()
    rows = build_pdf_report_rows(model_dict)
    diagram = build_pdf_report_diagram(model_dict)

    data["logo"] = logo
    data["bgd"] = bgd
    data["row"] = rows
    data["diagram"] = diagram
    data["print_date"] = datetime.datetime.now().strftime("%d/%m/%Y, %H:%M:%S")

    render = render_template("top_view_report.html", datas=data)

    pdf_file = pdfkit.from_string(render, False, options=options)

    return pdf_file
