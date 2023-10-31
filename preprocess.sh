#!/bin/bash -eu
IMAGE=diffusion-pre
docker build -t $IMAGE -f Dockerfile.preprocess . 

MOUNT=/mnt/sd/torch/torch-diffusion
MOUNT_DATA=/mnt/sd/torch/data_2
source local.env.sh
sudo docker run -it --rm --runtime nvidia \
 --shm-size=8G \
 --memory=20489m \
 --cpus=6 \
 --network host \
  -e NEPTUNE_API_KEY="${NEPTUNE_API_KEY}" \-v ${MOUNT_DATA}:/opt/data -v ${MOUNT}:/root/workspace $IMAGE 


