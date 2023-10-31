import os
from typing import List
import torchvision.transforms as transforms
import torchvision
from tqdm import tqdm
from PIL import Image


class PreProcessor:
    data_dir: str
    output_dir: str
    image_files: List[str]
    target_height: int
    target_width: int

    def __init__(
        self,
        data_dir="../data",
        output_dir="./preprocessed_data",
        target_width=128,
        target_height=192,
    ) -> None:
        self.target_height = target_height
        self.target_width = target_width
        self.data_dir = data_dir
        self.output_dir = os.path.join(output_dir, f"{target_width}x{target_height}")
        os.makedirs(output_dir, exist_ok=True)
        self.image_files = [
            os.path.join(self.data_dir, fname) for fname in os.listdir(self.data_dir)
        ]

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

        img = transform(img=img)

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

                # Extract the filename (without extension) from the path
                filename = os.path.splitext(os.path.basename(image_path))[0]

                # Save the transformed image to the output directory
                output_path = os.path.join(self.output_dir, f"{filename}.jpg")
                torchvision.utils.save_image(transformed_image, output_path)
                # Update the progress bar every 10 iterations
                if i % 10 == 0:
                    pbar.update(10)

        print("Transformed images saved to:", self.output_dir)
