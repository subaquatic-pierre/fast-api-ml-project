import json
import os
from flask import current_app, render_template
from flask import Blueprint
import requests

main = Blueprint("main", __name__)


def api_url():
    url = current_app.config.get("API_URL")
    return url


@main.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@main.route("/login", methods=["GET"])
def login():
    return render_template("login.html")


@main.route("/user", methods=["GET"])
def list_users():
    res = requests.get(f"{api_url()}/user")
    users = json.loads(res.text).get("data")
    return render_template("users.html", data=users)


@main.route("/case", methods=["GET"])
def list_cases():
    res = requests.get(f"{api_url()}/case")
    cases = json.loads(res.text).get("data", [])
    return render_template("cases.html", data=cases)


@main.route("/user/<user_id>", methods=["GET"])
def get_user(user_id):
    # Get user
    user_res = requests.get(f"{api_url()}/user/{user_id}")
    user = json.loads(user_res.text).get("data")

    # Get user API keys
    api_keys_res = requests.get(f"{api_url()}/api-key?userId={user_id}")
    api_keys = json.loads(api_keys_res.text).get("data", [])

    # Get user cases
    cases_res = requests.get(f"{api_url()}/case?userId={user_id}")
    cases = json.loads(cases_res.text).get("data", [])

    return render_template(
        "user.html", data={"user": user, "api_keys": api_keys, "cases": cases}
    )


@main.route("/case/<case_id>", methods=["GET"])
def get_case(case_id):

    # # Get user
    case_res = requests.get(f"{api_url()}/case/{case_id}")
    case = json.loads(case_res.text).get("data")

    return render_template("case.html", data=case)


@main.route("/register", methods=["GET"])
def register():
    return render_template("register.html")


@main.route("/new-case", methods=["GET"])
def damage_assessment():
    return render_template("new-case.html")


@main.route("/case/<case_id>/add-vehicle", methods=["GET"])
def add_vehicle(case_id):
    return render_template("add-vehicle.html")
