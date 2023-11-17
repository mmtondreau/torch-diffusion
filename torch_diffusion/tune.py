from ray.train.lightning import (
    RayDDPStrategy,
    RayLightningEnvironment,
    RayTrainReportCallback,
    prepare_trainer,
)
import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.train import RunConfig, ScalingConfig, CheckpointConfig
from ray.train.torch import TorchTrainer

import pytorch_lightning as pl
import torch
from torch_diffusion.data.image_data_module import ImageDataModule
from torch_diffusion.model.context_unit import ContextUnitConfig, ContextUnitLayerCOnfig
from torch_diffusion.model.difussion_model import DiffusionModule, DiffusionModuleConfig

# from lightning.pytorch.loggers import NeptuneLogger

import itertools

torch.distributed.init_process_group("gloo")
ray.init(num_cpus=6, num_gpus=1)


def get_permutations():
    # Define the possible features and kernel sizes
    features = [64, 128, 256, 512, 1024]
    kernel_sizes = [3, 5, 9]

    # Generate all combinations of features and kernel sizes
    all_combinations = list(itertools.product(features, kernel_sizes))

    # Function to check if the features are non-decreasing and kernel sizes are non-increasing
    def is_valid_permutation(permutation):
        return all(
            permutation[i][0] < permutation[i + 1][0]
            for i in range(len(permutation) - 1)
        ) and all(
            permutation[i][1] > permutation[i + 1][1]
            for i in range(len(permutation) - 1)
        )

    # Function to generate and filter permutations of a given length
    def generate_filtered_permutations(length):
        perms = itertools.permutations(all_combinations, length)
        return [perm for perm in perms if is_valid_permutation(perm)]

    # Generate and filter permutations for lengths 2, 3, and 4
    permutations_length_2 = generate_filtered_permutations(2)
    permutations_length_3 = generate_filtered_permutations(3)
    permutations_length_4 = generate_filtered_permutations(4)

    # Combine all filtered permutations
    return permutations_length_2 + permutations_length_3 + permutations_length_4


def train_func(config):
    dm = ImageDataModule(
        batch_size=128,
        num_workers=2,
        validation_split=0.2,
        test_split=0.1,
        data_dir="./preprocessed_dir",
        width=32,
        height=48,
        truncate=1000,
    )
    model = DiffusionModule(
        model_config=ContextUnitConfig(
            features=config["features"],
            layers=[
                ContextUnitLayerCOnfig(features=f, kernel_size=ks)
                for f, ks in config["layers"]
            ],
        ),
        config=DiffusionModuleConfig(height=48, width=32),
    )

    trainer = pl.Trainer(
        devices="auto",
        accelerator="auto",
        strategy=RayDDPStrategy(),
        callbacks=[RayTrainReportCallback()],
        plugins=[RayLightningEnvironment()],
        enable_progress_bar=False,
        gradient_clip_val=0.5,
        accumulate_grad_batches=4,
    )
    trainer = prepare_trainer(trainer)
    trainer.fit(model, datamodule=dm)


if __name__ == "__main__":
    search_space = {
        "features": tune.choice([64, 128, 256]),
        "layers": tune.choice(get_permutations()),
    }
    # The maximum training epochs
    num_epochs = 5

    # Number of sampls from parameter space
    num_samples = 10

    scheduler = ASHAScheduler(max_t=num_epochs, grace_period=1, reduction_factor=2)

    scaling_config = ScalingConfig(
        num_workers=1, use_gpu=True, resources_per_worker={"GPU": 1, "CPU": 4}
    )

    run_config = RunConfig(
        checkpoint_config=CheckpointConfig(
            num_to_keep=2,
            checkpoint_score_attribute="val/epoch/loss",
            checkpoint_score_order="min",
        ),
    )

    # Define a TorchTrainer without hyper-parameters for Tuner
    ray_trainer = TorchTrainer(
        train_func,
        scaling_config=scaling_config,
        run_config=run_config,
    )

    def tune_asha(num_samples=10):
        scheduler = ASHAScheduler(
            max_t=num_epochs, grace_period=2, reduction_factor=3, brackets=3
        )

        tuner = tune.Tuner(
            ray_trainer,
            param_space={"train_loop_config": search_space},
            tune_config=tune.TuneConfig(
                metric="val/epoch/loss",
                mode="min",
                num_samples=num_samples,
                scheduler=scheduler,
            ),
        )
        return tuner.fit()

    results = tune_asha(num_samples=num_samples)
