from time import sleep, time
from utils import timer

from utils import check_all_case_status, get_case_ids
from make_requests import make_list_cases_request, make_new_case_request

from utils import get_case_ids

from make_requests import (
    make_delete_cases_request,
    make_list_cases_request,
    make_new_case_request,
    make_add_vehicle_request,
)

from log import logger


def delete_all_cases():
    case_list = make_list_cases_request()
    case_id_list = get_case_ids(case_list)
    delete_cases_response_data = make_delete_cases_request(case_id_list)
    return delete_cases_response_data


@timer
def add_cars(case_ids):
    print(f"Number of cases to add cars to {len(case_ids)}")
    print("Adding 4 cars to each case...")
    data = make_add_vehicle_request(case_ids)
    print("Cars added to cases")
    return data


@timer
def create_cases(num_cases):
    msg = "Create new cases"
    print(msg)
    logger.info(msg)

    cases = make_new_case_request(num_cases)

    msg = "New cases created"
    print(msg)
    logger.info(msg)

    # Get IDs for all new cases
    # all_cases = make_list_cases_request()
    case_ids = get_case_ids(cases)

    # Ensure all cases initialized with empty report
    all_report_generated = False
    while not all_report_generated:
        msg = f"Polling for {case_ids} reportGenerated == True"
        logger.info(msg)
        print(msg)

        if len(case_ids) == 0:
            all_report_generated = True
        else:
            status = check_all_case_status(case_ids, "reportGenerated", True)
            all_report_generated = status

        sleep(1)

    msg = "All case reports generated"
    logger.info(msg)
    print(msg)

    return case_ids


def start_machine_learning_worker(case_ids):
    log = f"Machine learning worker started with {len(case_ids)} cases"
    print(log)
    logger.info(log)

    t1 = time()
    # Add cars to cases, start machine learning worker
    add_cars(case_ids)

    # Start polling server to check case status of each case
    cars_added = False
    while not cars_added:
        print(f"Polling for {len(case_ids)} cases...")
        if len(case_ids) == 0:
            cars_added = True
        else:
            status = check_all_case_status(case_ids, "status", "CAR_ADDED")
            cars_added = status
            sleep(1)

    t2 = time()
    log = f"Machine learning completed for {len(case_ids)} cases in {(t2-t1):.2f}s"
    print(log)
    logger.info(log)
