#!/bin/bash -eu
IMAGE=diffusion
docker build -t $IMAGE -f Dockerfile . 

MOUNT=/mnt/sd/torch/torch-diffusion
source local.env.sh
sudo docker run -it --rm --runtime nvidia --shm-size=8G --memory-swap=20489m --memory=20489m --cpus=6 --network host -e NEPTUNE_API_KEY="${NEPTUNE_API_KEY}" -v ${MOUNT}:/root/workspace $IMAGE 


