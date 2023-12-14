from PIL import Image
from torch.autograd import Variable
from torchvision import transforms
from torch import nn
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


def image_sort(dir):
    image_data = pd.read_csv('facial_expressions/data/legend.csv')
    emotions_seen = set()

    for root, dirs, files in os.walk(dir):
        for img in files:
            error = False
            file_path = os.path.join(root, img)
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


def create_testing_set(original_folder, testing_folder, num_images_to_move):
    os.makedirs(testing_folder, exist_ok=True)
    all_images = []
    for emotion_folder in os.listdir(original_folder):
        emotion_path = os.path.join(original_folder, emotion_folder)

        if not os.path.isdir(emotion_path):
            continue

        emotion_images = os.listdir(emotion_path)
        all_images.extend([(emotion_folder, image)
                          for image in emotion_images])

    selected_images = random.sample(all_images, num_images_to_move)

    for emotion, image in selected_images:
        source_path = os.path.join(original_folder, emotion, image)
        destination_path = os.path.join(testing_folder, emotion, image)
        os.makedirs(os.path.dirname(destination_path), exist_ok=True)
        shutil.move(source_path, destination_path)


def transform_data(organized_data_folder):
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    custom_dataset = datasets.ImageFolder(
        root=organized_data_folder, transform=transform)

    batch_size = 64
    custom_dataloader = DataLoader(
        custom_dataset, batch_size=batch_size, shuffle=True)

    print(f"Number of samples in the custom dataset: {len(custom_dataset)}")

    for inputs, labels in custom_dataloader:
        print(f"Batch shape: {inputs.shape}, Labels shape: {labels.shape}")
        break

    return (custom_dataset, custom_dataloader)


class NeuralNetwork(nn.Module):
    def __init__(self, num_classes):
        super(NeuralNetwork, self).__init__()
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


def train_model(custom_dataset, custom_dataloader):
    num_classes = len(custom_dataset.classes)
    model = NeuralNetwork(num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 1000
    for epoch in range(num_epochs):
        total_loss = 0.0

        for inputs, labels in custom_dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(labels)

        average_loss = total_loss / \
            len(custom_dataset)
        print(
            f'Epoch [{epoch+1}/{num_epochs}], Average Training Loss: {average_loss}')

    torch.save(model.state_dict(), 'emotion_model.pth')


def count_images_per_folder(dataset_folder):
    image_counts = {}
    total = 0
    for emotion_folder in os.listdir(dataset_folder):
        emotion_path = os.path.join(dataset_folder, emotion_folder)

        if not os.path.isdir(emotion_path):
            continue

        emotion_images = os.listdir(emotion_path)
        image_counts[emotion_folder] = len(emotion_images)
        total += len(emotion_images)

    return total


def load_and_predict(model_path, test_images_path):
    num_classes = 8
    model = NeuralNetwork(num_classes)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    test_data = []
    true_labels = []
    class_labels = sorted(os.listdir(test_images_path))

    for label_id, label in enumerate(class_labels):
        label_path = os.path.join(test_images_path, label)

        for image_name in os.listdir(label_path):
            image_path = os.path.join(label_path, image_name)

            img = Image.open(image_path).convert('RGB')
            img = transform(img)
            test_data.append(img)
            true_labels.append(label_id)

    test_data = torch.stack(test_data)
    with torch.no_grad():
        outputs = model(test_data)

    _, predicted_labels = torch.max(outputs, 1)

    correct_predictions = (
        predicted_labels == torch.tensor(true_labels)).sum().item()
    total_samples = len(true_labels)
    accuracy = correct_predictions / total_samples * 100

    # Print accuracy
    print(f'Correct predictions: {correct_predictions}/{total_samples}')
    print(f'Accuracy: {accuracy:.2f}%')
    return predicted_labels


if __name__ == "__main__":
    # image_sort("facial_expressions/images")
    train_set = 'facial_expressions/train'
    testing_set = 'facial_expressions/test'
    # create_testing_set(original_set, testing_set, 2052)
    # create_testing_set(original_set, validation_set, 2052)
    # print(f"OG: {count_images_per_folder(train_set)}")
    # print(f"Test: {count_images_per_folder(testing_set)}")
    # print(f"Validation: {count_images_per_folder(validation_set)}")
    # custom_dataset, custom_dataloader = transform_data(train_set)
    # train_model(custom_dataset, custom_dataloader)
    labels = load_and_predict('facial_expressions/emotion_model.pth',
                              'facial_expressions/images_validation')
