import os
from shutil import copyfile
import shutil
import random
import pandas as pd
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim

# Function to sort images based on emotions


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

# Function to transform data


def create_testing_set(original_folder, testing_folder, num_images_to_move):
    # Create the testing folder if it doesn't exist
    os.makedirs(testing_folder, exist_ok=True)

    # Get a list of all images in the original dataset
    all_images = []
    for emotion_folder in os.listdir(original_folder):
        emotion_path = os.path.join(original_folder, emotion_folder)

        # Skip non-directory files in the original dataset
        if not os.path.isdir(emotion_path):
            continue

        emotion_images = os.listdir(emotion_path)
        all_images.extend([(emotion_folder, image)
                          for image in emotion_images])

    # Randomly select a subset of images
    selected_images = random.sample(all_images, num_images_to_move)

    # Move selected images to the testing folder
    for emotion, image in selected_images:
        source_path = os.path.join(original_folder, emotion, image)
        destination_path = os.path.join(testing_folder, emotion, image)
        os.makedirs(os.path.dirname(destination_path), exist_ok=True)
        shutil.move(source_path, destination_path)


def transform_data():
    # Define data transformations
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Path to the folder containing organized data
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

    return (custom_dataset, custom_dataloader)
# Function to train a simple neural network


def train_model(custom_dataset, custom_dataloader):
    # Define a simple neural network
    class SimpleNet(nn.Module):
        def __init__(self, num_classes):
            super(SimpleNet, self).__init__()
            self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
            self.relu = nn.ReLU()
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
            self.fc1 = nn.Linear(64 * 32 * 32, num_classes)

        def forward(self, x):
            x = self.conv1(x)
            x = self.relu(x)
            x = self.pool(x)
            x = x.view(x.size(0), -1)
            x = self.fc1(x)
            return x

    # Initialize the neural network
    num_classes = len(custom_dataset.classes)
    model = SimpleNet(num_classes)

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    num_epochs = 10

    for epoch in range(num_epochs):
        total_loss = 0.0  # Initialize total loss for the epoch

        for inputs, labels in custom_dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(labels)  # Accumulate the total loss

        average_loss = total_loss / \
            len(custom_dataset)  # Calculate average loss
        print(
            f'Epoch [{epoch+1}/{num_epochs}], Average Training Loss: {average_loss}')

    # Save the trained model
    torch.save(model.state_dict(), 'emotion_model.pth')


def count_images_per_folder(dataset_folder):
    image_counts = {}
    total = 0
    for emotion_folder in os.listdir(dataset_folder):
        emotion_path = os.path.join(dataset_folder, emotion_folder)

        # Skip non-directory files in the dataset
        if not os.path.isdir(emotion_path):
            continue

        emotion_images = os.listdir(emotion_path)
        image_counts[emotion_folder] = len(emotion_images)
        total += len(emotion_images)

    return total


# Main execution
if __name__ == "__main__":
    # Replace 'your_image_folder' with the actual path to your image folder
    # image_sort(image_folder)
    original_set = 'facial_expressions/images_sorted'
    testing_set = 'facial_expressions/images_test'
    validation_set = 'facial_expressions/images_validation'
    # create_testing_set(original_set, testing_set, 2052)
    # create_testing_set(original_set, validation_set, 2052)
    print(f"OG: {count_images_per_folder(original_set)}")
    print(f"Test: {count_images_per_folder(testing_set)}")
    print(f"Validation: {count_images_per_folder(validation_set)}")
    custom_dataset, custom_dataloader = transform_data()
    train_model(custom_dataset, custom_dataloader)
