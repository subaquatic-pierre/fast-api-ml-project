# Accident AI App

## Elastic IP addresses

* API_URL: 34.233.11.37
* WORKER_URL: 34.193.150.251
* MONGODB_URL: 44.206.194.61

## Detectron

* python -m pip install wheel
* python -m pip install torchvision
* python -m pip install opencv-python
* python -m pip install 'git+<https://github.com/facebookresearch/detectron2.git>'

## Get local IP

hostname -I | awk '{print $1}'
