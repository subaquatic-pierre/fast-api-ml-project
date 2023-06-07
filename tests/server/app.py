from time import sleep
from flask import Flask, jsonify, request

app = Flask(__name__)


@app.route("/")
def root():
    return jsonify({"status": True})


@app.route("/api/case/<string:case_id>/add-vehicle", methods=["POST"])
def add_vehicle(case_id):
    data = request.json
    return jsonify({"data": data})


@app.route("/api/case", methods=["POST"])
def case():
    data = request.json
    sleep(2)
    return jsonify({"data": data})
