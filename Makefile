
TRAINING_IMAGE=diffusion-training
PREPROCESS_IMAGE=diffusion-preprocess
INFERENCE_IMAGE=diffusion-inference
RDOCKER_REGISTRY=registry.tonberry.org/tonberry
VERSION=latest
TRAINING_JOB_NAME=diffusion-training-job
PREPROCESS_JOB_NAME=diffusion-preprocess-job
INFERENCE_JOB_NAME=diffusion-inference-job

.PHONY: build-training
build-training: 
	docker build -t ${TRAINING_IMAGE}:${VERSION} -f Dockerfile.training .

.PHONY: tag-training
tag-training: build-training
	docker tag ${TRAINING_IMAGE}:${VERSION} ${RDOCKER_REGISTRY}/${TRAINING_IMAGE}:${VERSION}

.PHONY: push-training
push-training: tag-training
	docker push ${RDOCKER_REGISTRY}/${TRAINING_IMAGE}:${VERSION}

.PHONY: delete-job-training
delete-job-training: 
	kubectl delete job ${TRAINING_JOB_NAME} --ignore-not-found=true 

.PHONY: submit-training
submit-training: delete-job-training push-training
	kubectl apply -f k8/common \
	&& kubectl apply -f k8/training \
	&& sleep 5 \
	&& kubectl get pods -l job-name=${TRAINING_JOB_NAME} -o json | jq -r '.items | sort_by(.metadata.creationTimestamp) | .[-1].metadata.name' | xargs kubectl logs -f

.PHONY: training-logs
training-logs:
	kubectl get pods -l job-name=${TRAINING_JOB_NAME} -o json | jq -r '.items | sort_by(.metadata.creationTimestamp) | .[-1].metadata.name' | xargs kubectl logs -f
	# kubectl get pods -l job-name=${TRAINING_JOB_NAME} | tail -n -1 | awk '{print $$1}' | xargs kubectl logs -f 


.PHONY: build-preprocess
build-preprocess: 
	docker build -t ${PREPROCESS_IMAGE}:${VERSION} -f Dockerfile.preprocess .

.PHONY: tag-preprocess
tag-preprocess: build-preprocess
	docker tag ${PREPROCESS_IMAGE}:${VERSION} ${RDOCKER_REGISTRY}/${PREPROCESS_IMAGE}:${VERSION}

.PHONY: push-preprocess
push-preprocess: tag-preprocess
	docker push ${RDOCKER_REGISTRY}/${PREPROCESS_IMAGE}:${VERSION}

.PHONY: delete-job-preprocess
delete-job-preprocess: 
	kubectl delete job ${PREPROCESS_JOB_NAME} --ignore-not-found=true 

.PHONY: submit-preprocess
submit-preprocess: delete-job-preprocess push-preprocess
	kubectl apply -f k8/common \
	&& kubectl apply -f k8/preprocess \
	&& sleep 5 \
	&& kubectl get pods -l job-name=${PREPROCESS_JOB_NAME} -o json | jq -r '.items | sort_by(.metadata.creationTimestamp) | .[-1].metadata.name' | xargs kubectl logs -f

.PHONY: preprocess-logs
preprocess-logs:
	kubectl get pods -l job-name=${PREPROCESS_JOB_NAME} -o json | jq -r '.items | sort_by(.metadata.creationTimestamp) | .[-1].metadata.name' | xargs kubectl logs -f
	# kubectl get pods -l job-name=${PREPROCESS_JOB_NAME} | tail -n -1 | awk '{print $$1}' | xargs kubectl logs -f 



.PHONY: build-inference
build-inference: 
	docker build -t ${INFERENCE_IMAGE}:${VERSION} -f Dockerfile.inference .

.PHONY: tag-inference
tag-inference: build-inference
	docker tag ${INFERENCE_IMAGE}:${VERSION} ${RDOCKER_REGISTRY}/${INFERENCE_IMAGE}:${VERSION}

.PHONY: push-inference
push-inference: tag-inference
	docker push ${RDOCKER_REGISTRY}/${INFERENCE_IMAGE}:${VERSION}

.PHONY: delete-job-inference
delete-job-inference: 
	kubectl delete job ${INFERENCE_JOB_NAME} --ignore-not-found=true 

.PHONY: submit-inference
submit-inference: delete-job-inference push-inference
	kubectl apply -f k8/common \
	&& kubectl apply -f k8/inference \
	&& sleep 5 \
	&& kubectl get pods -l job-name=${INFERENCE_JOB_NAME} -o json | jq -r '.items | sort_by(.metadata.creationTimestamp) | .[-1].metadata.name' | xargs kubectl logs -f

.PHONY: inference-logs
inference-logs:
	kubectl get pods -l job-name=${INFERENCE_JOB_NAME} -o json | jq -r '.items | sort_by(.metadata.creationTimestamp) | .[-1].metadata.name' | xargs kubectl logs -f
	# kubectl get pods -l job-name=${INFERENCE_JOB_NAME} | tail -n -1 | awk '{print $$1}' | xargs kubectl logs -f 

