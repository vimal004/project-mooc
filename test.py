import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt


# Define the model architecture (same as during training)
class ResNet9(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.conv1 = ConvBlock(in_channels, 64)
        self.conv2 = ConvBlock(64, 128, pool=True)
        self.res1 = nn.Sequential(ConvBlock(128, 128), ConvBlock(128, 128))
        self.conv3 = ConvBlock(128, 256, pool=True)
        self.conv4 = ConvBlock(256, 512, pool=True)
        self.res2 = nn.Sequential(ConvBlock(512, 512), ConvBlock(512, 512))
        self.classifier = nn.Sequential(
            nn.MaxPool2d(4), nn.Flatten(), nn.Linear(512, num_classes)
        )

    def forward(self, xb):
        out = self.conv1(xb)
        out = self.conv2(out)
        out = self.res1(out) + out
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out) + out
        out = self.classifier(out)
        return out


# Define the ConvBlock (used in ResNet9)
def ConvBlock(in_channels, out_channels, pool=False):
    layers = [
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    ]
    if pool:
        layers.append(nn.MaxPool2d(4))
    return nn.Sequential(*layers)


# Load the entire model (not just the state_dict)
model_path = "./plant-disease-model-complete.pth"  # Update this path if necessary
model = torch.load(
    model_path, map_location=torch.device("cpu")
)  # Load the entire model

# Set the model to evaluation mode
model.eval()

# Define the class names (replace with your actual class names)
class_names = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Blueberry___healthy",
    "Cherry_(including_sour)___Powdery_mildew",
    "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot",
    "Peach___healthy",
    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Raspberry___healthy",
    "Soybean___healthy",
    "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch",
    "Strawberry___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy",
]


# Function to preprocess the image
def preprocess_image(image_path):
    transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),  # Resize to match training size
            transforms.ToTensor(),  # Convert to tensor
        ]
    )
    image = Image.open(image_path).convert("RGB")  # Open image and convert to RGB
    image = transform(image).unsqueeze(0)  # Add batch dimension
    return image


# Function to predict the class of an image
def predict_image(image_path, model, class_names):
    image = preprocess_image(image_path)  # Preprocess the image
    with torch.no_grad():  # Disable gradient calculation
        output = model(image)  # Get model predictions
    _, predicted_idx = torch.max(output, 1)  # Get the predicted class index
    predicted_class = class_names[predicted_idx.item()]  # Get the class name
    return predicted_class


# Path to the test image
test_image_path = "peach.jpg"  # Replace with the path to your test image

# Predict the class of the test image
predicted_class = predict_image(test_image_path, model, class_names)
print(f"Predicted Class: {predicted_class}")

# Display the test image
image = Image.open(test_image_path)
plt.imshow(image)
plt.title(f"Predicted: {predicted_class}")
plt.axis("off")
plt.show()
