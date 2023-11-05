import os
from typing import List
import torchvision.transforms as transforms
from tqdm import tqdm
from PIL import Image
import torch


class PreProcessor:
    data_dir: str
    output_dir: str
    image_files: List[str]
    target_height: int
    target_width: int
    _samples_per_image: int

    def __init__(
        self,
        data_dir="../data",
        output_dir="../preprocessed_data",
        target_width=128,
        target_height=192,
    ) -> None:
        self._samples_per_image = 3
        self.target_height = target_height
        self.target_width = target_width
        self.data_dir = data_dir
        self.output_dir = os.path.join(output_dir, f"{target_width}x{target_height}")
        os.makedirs(self.output_dir, exist_ok=True)
        self.image_files = [
            os.path.join(self.data_dir, fname) for fname in os.listdir(self.data_dir)
        ]

        self._timesteps = 500
        self._beta1 = 1e-4
        self._beta2 = 0.02
        # construct DDPM noise schedule
        self.b_t = (self._beta2 - self._beta1) * torch.linspace(
            0, 1, self._timesteps + 1
        ) + self._beta1
        self.a_t = 1 - self.b_t
        self.ab_t = torch.cumsum(self.a_t.log(), dim=0).exp()
        self.ab_t[0] = 1

    def transform(self, img):
        # Get the width and height of the image
        width, height = img.size

        # Check if the image is in landscape orientation
        if width > height:
            # Rotate the image by 90 degrees
            img = img.transpose(Image.Transpose.ROTATE_90)

        # Apply the other transformations
        transform = transforms.Compose(
            [
                transforms.Resize(
                    (self.target_height, self.target_width)
                ),  # Resize the image
                transforms.ToTensor(),  # Convert the image to a PyTorch tensor
            ]
        )

        transform2 = transforms.Compose(
            [
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # range [-1,1]
            ]
        )

        img = transform(img=img)
        img = transform2(img=img)
        return img

    def process(self):
        with tqdm(
            total=len(self.image_files), desc="Processing images", unit="image"
        ) as pbar:
            # Loop through the image files and apply transformations
            for i, image_path in enumerate(self.image_files):
                # Open the image
                image = Image.open(image_path).convert("RGB")

                # Apply transformations
                transformed_image = self.transform(image)
                for sample_num in range(0, self._samples_per_image):
                    perturbed_image, noise, t = self._perturb_input(transformed_image)
                    # Extract the filename (without extension) from the path
                    filename = os.path.splitext(os.path.basename(image_path))[0]
                    # Save the transformed image to the output directory
                    output_path = os.path.join(
                        self.output_dir, f"{filename}.{sample_num}.pt"
                    )
                    torch.save((perturbed_image, noise, t), output_path)
                    # Update the progress bar every 10 iterations
                if i % 10 == 0:
                    pbar.update(10)

        print("Transformed images saved to:", self.output_dir)

    def _perturb_input(self, x):
        noise = torch.randn_like(x)
        t = torch.randint(1, self._timesteps + 1, (1,))
        # print(f"t device: {t.device}, x device: {x.device}, ab_t device: {self.ab_t.device}")
        return (
            (
                self.ab_t.sqrt()[t, None, None] * x
                + (1 - self.ab_t[t, None, None]) * noise
            ),
            noise,
            t,
        )
