FROM registry.tonberry.org/tonberry/torch:1.0
ARG PYTHON_MODULE=torch_diffusion
RUN conda run -n torch pip install neptune
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "torch", "python", "-m", "torch_diffusion"]
