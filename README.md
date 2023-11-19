# FastAPI ML Project

## Overview

FastAPI ML Project is a machine learning application that generates damage assessment reports based on images of car accidents. The project utilizes FastAPI for efficient web API development and incorporates an asynchronous worker node to make predictions. This architecture enables users to receive quick responses, with predictions being updated once processing is complete.

## Features

- **FastAPI Framework**: Utilizes the FastAPI framework for building robust and fast web APIs.
- **Asynchronous Prediction**: Implements a separate worker node for asynchronous prediction, allowing users to receive initial responses promptly.
- **Damage Assessment**: Analyzes images of car accidents to generate detailed damage assessment reports.

## Getting Started

### Prerequisites

- [Docker](https://www.docker.com/)
- [Docker Compose](https://docs.docker.com/compose/)

### Usage with Docker Compose

1. Clone the repository:

git clone https://github.com/subaquatic-pierre/fast-api-ml-project.git

arduino

2. Build and run the Docker containers:

cd fast-api-ml-project
docker-compose up --build

bash

3. Access the API at http://127.0.0.1:8000/docs for interactive documentation.

## Asynchronous Prediction

To run the asynchronous worker node for predictions, execute the following command:

python worker.py

This ensures that predictions are processed in the background, and users receive updates once the assessment is complete.

## Contributing

Contributions are welcome! Feel free to open issues or submit pull requests.

## License

This project is licensed under the [MIT License](LICENSE).
