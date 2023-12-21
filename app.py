import streamlit as st
import torch
import torch.nn as nn
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from PIL import Image
import time
import torch.nn.functional as F

# Define the SimpleCNN architecture
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(in_features=32 * 56 * 56, out_features=128)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(in_features=128, out_features=num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        return x

# Mapping from class index to class name
class_index_to_name = {
    0: 'Class Belimbing Wulu',
    1: 'Class Jambu Biji',
    2: 'Class Jeruk Nipis',
    3: 'Class Kemangi',
    4: 'Class Lidah Buaya',
    5: 'Class Nangka',
    6: 'Class Pandan',
    7: 'Clas Pepaya',
    8: 'Class Seledri',
    9: 'Class Sirih'
    # Add more class mappings as needed
}

# Create the Streamlit app
def main():
    st.title("Leaf Classification App")

    # Upload image through Streamlit
    uploaded_file = st.file_uploader("Choose a leaf image...")

    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image.", use_column_width=True)

        # Load the pre-trained model
        model = SimpleCNN(num_classes=10)  # Change the number of classes accordingly
        model_path = "model.h5"  # Change the path accordingly
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()

        # Define the image transformations
        transform = Compose([
            Resize((224, 224)),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Preprocess the uploaded image
        input_tensor = transform(image)
        input_batch = input_tensor.unsqueeze(0)  # Add a batch dimension

        # Measure execution time
        start_time = time.time()

        # Make predictions
        with torch.no_grad():
            output = model(input_batch)

        # Calculate execution time
        execution_time = time.time() - start_time

        # Apply softmax to get probabilities
        probabilities = F.softmax(output[0], dim=0)

        # Display the prediction results
        class_index = torch.argmax(output[0]).item()
        class_name = class_index_to_name.get(class_index, f'Unknown Class {class_index}')
        confidence = probabilities[class_index].item() * 100
        st.write(f"Prediction: {class_name}")
        st.write(f"Confidence: {confidence:.2f}%")
        st.write(f"Execution Time: {execution_time:.4f} seconds")

if __name__ == "__main__":
    main()
