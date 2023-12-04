from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import torch
import os
import pandas as pd
from shutil import copyfile


def image_sort(dir):
    image_data = pd.read_csv('facial_expressions/data/legend.csv')
    emotions_seen = set()

    for root, dirs, files in os.walk(dir):
        for img in files:
            error = False
            file_path = os.path.join(root, img)

            # Obtain the class for the image from the CSV file
            file_name = os.path.basename(file_path)
            try:
                class_label = image_data.loc[image_data['image']
                                             == file_name, 'emotion'].values[0]
            except IndexError:
                error = True

            if not error:
                if class_label not in emotions_seen:
                    emotion_folder = os.path.join(
                        'facial_expressions/images_sorted', str(class_label))
                    os.makedirs(emotion_folder, exist_ok=True)

                destination_path = os.path.join(emotion_folder, file_name)
                copyfile(file_path, destination_path)


def transform_data():
    # Define data transformations
    transform = transforms.Compose([
        transforms.Resize((64, 64)),  # Resize images to a consistent size
        transforms.ToTensor(),         # Convert images to tensors
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
                             )  # Normalize pixel values
    ])

    # Path to the folder containing your organized data
    organized_data_folder = 'facial_expressions/images_sorted'

    # Create a PyTorch dataset using ImageFolder
    custom_dataset = datasets.ImageFolder(
        root=organized_data_folder, transform=transform)

    # Create a PyTorch DataLoader
    batch_size = 64
    custom_dataloader = DataLoader(
        custom_dataset, batch_size=batch_size, shuffle=True)

    # Check the size of the custom dataset
    print(f"Number of samples in the custom dataset: {len(custom_dataset)}")

    # Iterate over the DataLoader to access batches of data
    for inputs, labels in custom_dataloader:
        # Your training or evaluation loop here
        print(f"Batch shape: {inputs.shape}, Labels shape: {labels.shape}")
        break  # Break after the first batch for demonstration purposes


def train_model():
    pass


if __name__ == "__main__":
    transform_data()
    # path = "facial_expressions/images"
    # image_sort(path)
