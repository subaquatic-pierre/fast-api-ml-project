from handlers import (
    create_cases,
    start_machine_learning_worker,
)
from utils import start_log, end_log


def main():
    start_log()

    # Generate 1 new
    case_ids = create_cases(1)

    start_machine_learning_worker(case_ids)

    end_log()


main()
