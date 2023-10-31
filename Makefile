
TRAINING_IMAGE=diffusion-training
RDOCKER_REGISTRY=registry.tonberry.org/tonberry
VERSION=latest
JOB_NAME=diffusion-job

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
	kubectl delete job ${JOB_NAME} --ignore-not-found=true 

.PHONY: submit-training
submit-training: delete-job-training push-training
	kubectl apply -f k8 \
	&& kubectl get pods -l job-name=diffusion-job -o json | jq -r '.items[] | select(.status.phase=="Running" or .status.phase=="Pending").metadata.name' | xargs kubectl logs -f

.PHONY: training-logs
training-logs:
	kubectl get pods -l job-name=diffusion-job -o json | jq -r '.items[] | select(.status.phase=="Running" or .status.phase=="Pending").metadata.name' | xargs kubectl logs -f
	# kubectl get pods -l job-name=diffusion-job | tail -n -1 | awk '{print $$1}' | xargs kubectl logs -f 
