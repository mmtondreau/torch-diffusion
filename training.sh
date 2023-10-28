#!/bin/bash -eu
IMAGE=diffusion-training
docker build -t $IMAGE -f Dockerfile.training . 

MOUNT=/mnt/sd/torch/torch-diffusion
source local.env.sh
sudo docker run -it --rm \
	--runtime nvidia \
	--shm-size=8G \
       	--memory-swap=20489m \
	--memory=20489m \
	--cpus=6 \
	--network host \
	-e NEPTUNE_API_KEY="${NEPTUNE_API_KEY}" \
	-e SLACK_TOKEN="${SLACK_TOKEN}" \
	-e SLACK_CHANNEL_MONITORING="${SLACK_CHANNEL_MONITORING}" \
	-v ${MOUNT}:/var/torch \
	$IMAGE 


