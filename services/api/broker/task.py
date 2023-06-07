from pymongo import MongoClient
from bson.objectid import ObjectId

from broker.worker import celery
from config.config import settings


@celery.task(name="new_task")
def create_task(target, *args):
    # Perfoem long run computation
    result = target(*args)

    # ---
    # Update database with result
    # ---

    # with MongoClient(settings.mongodb_host) as client:
    #     db = client[settings.mongodb_db]

    #     case = db.case.find_one({"_id": ObjectId(case_id)})

    #     old_vehicle_count = case["vehicleCount"]
    #     new_vehicle_count = old_vehicle_count + 1

    #     new_data = {
    #         "reportUrlPdf": None,
    #         "vehicleCount": new_vehicle_count,
    #         "reportUrlPdfTopView": None,
    #         "status": case_status,
    #     }

    #     case = db.case.update_one({"_id": ObjectId(case_id)}, {"$set": new_data})
