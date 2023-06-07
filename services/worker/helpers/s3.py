import logging
import os
import json
import requests

import boto3
from botocore.exceptions import ClientError

from config.config import settings
from helpers.utils import get_case_dir, clean_case_dir
from config.logger import logger


def get_s3_config():
    AWS_PUBLIC_KEY = settings.aws_public_key
    AWS_SECRET_KEY = settings.aws_secret_key
    BUCKET_NAME = settings.bucket_name

    return AWS_PUBLIC_KEY, AWS_SECRET_KEY, BUCKET_NAME


def upload_file(file_name, object_name=None):
    """Upload a file to an S3 bucket

    :param file_name: File to upload
    :param bucket: Bucket to upload to
    :param object_name: S3 object name. If not specified then file_name is used
    :return: True if file was uploaded, else False
    """
    AWS_PUBLIC_KEY, AWS_SECRET_KEY, BUCKET_NAME = get_s3_config()

    # If S3 object_name was not specified, use file_name
    if object_name is None:
        object_name = os.path.basename(file_name)

    # Upload the file
    s3_client = boto3.client(
        "s3", aws_access_key_id=AWS_PUBLIC_KEY, aws_secret_access_key=AWS_SECRET_KEY
    )

    logger.info(f"Uploading file {file_name}")
    try:
        response = s3_client.upload_file(file_name, BUCKET_NAME, object_name)
    except ClientError as e:
        logging.error(e)
        return False
    return True


def upload_case_to_s3(case_id):
    logger.info("Uploading files...")
    case_dir = get_case_dir(case_id)

    for filename in os.listdir(case_dir):
        file_key = f"{case_id}/{filename}"
        file_path = os.path.join(case_dir, filename)
        upload_file(file_path, file_key)

    logger.info("Upload complete")


def download_file(file_key, local_path):
    AWS_PUBLIC_KEY, AWS_SECRET_KEY, BUCKET_NAME = get_s3_config()

    print("Downloading file ...")
    print(file_key)

    # Download file
    s3_client = boto3.client(
        "s3", aws_access_key_id=AWS_PUBLIC_KEY, aws_secret_access_key=AWS_SECRET_KEY
    )
    try:
        with open(local_path, "wb") as file:
            response = s3_client.download_fileobj(BUCKET_NAME, file_key, file)
    except ClientError as e:
        logging.error(e)
        return False
    return True


def get_case_report(report_url: str) -> dict:
    report_res = requests.get(report_url)
    report = json.loads(report_res.content)
    return report
