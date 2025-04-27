---
layout: post
title: Coffee Bean Quality Classifier
---

## Coffee Bean Quality Classifier

This past semester, I undertook a personal project aimed at automating the process of grading coffee bean quality using computer vision. The goal was to build an image classification model capable of distinguishing between good beans and those with common defects. I decided to use **PyTorch** as the primary framework for model development, drawn by its flexibility and the wealth of online resources available.

Getting started with the dataset was the first major hurdle. I couldn't find a readily available, high-quality public dataset specifically for coffee bean defects, so I had to assemble my own. This involved collecting images of different types of beans, both healthy and defective (e.g., broken, insect-damaged, discolored). This data collection and initial labeling phase took roughly two weeks of sporadic work, mostly evenings and weekends. Standardizing the images – resizing, cropping, and ensuring a relatively consistent background – was tedious. I quickly realized inconsistent lighting was going to be a problem, something I had to just live with for this phase but noted as a potential area for improvement.

For the model, I opted to fine-tune a pre-trained convolutional neural network (CNN) from the torchvision library. Transfer learning seemed like the most practical approach given the limited size of my custom dataset compared to datasets like ImageNet. I initially tried a ResNet18, but after some early experiments that weren't converging as well as I hoped, I switched to a ResNet50. My hypothesis was that the slightly deeper architecture might capture more complex features relevant to subtle bean defects.

Training setup involved defining a standard cross-entropy loss and using the Adam optimizer. The training loop itself was fairly standard, but I spent a significant amount of time (probably 3-4 days total, including debugging) wrestling with data loading and augmentation pipelines. Getting the custom dataset integrated correctly with PyTorch's `Dataset` and `DataLoader` classes had a few snags. I remember hitting a persistent `PIL.UnidentifiedImageError` that turned out to be caused by some corrupt images I hadn't filtered out properly during collection. Debugging that involved adding more robust error handling and file validation checks during data loading.

```python
# Snippet showing part of the custom dataset class
class CoffeeBeanDataset(Dataset):
    def __init__(self, img_dir, annotations_file, transform=None):
        # annotations_file format: img_path,label
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        # Basic check for image validity - added after that PIL error
        self.valid_indices = [i for i in range(len(self.img_labels))
                              if self._is_valid_image(self.img_labels.iloc[i, 0])]

    def _is_valid_image(self, img_name):
        try:
            img_path = os.path.join(self.img_dir, img_name)
            img = Image.open(img_path)
            img.verify() # Check file integrity
            return True
        except (IOError, SyntaxError) as e:
            print(f"Bad file: {img_name}, {e}") # Log bad files
            return False
        finally:
            # Need to explicitly close in verify() cases
            if 'img' in locals() and img:
                img.close()


    def __len__(self):
        return len(self.valid_indices) # Use valid indices count

    def __getitem__(self, idx):
        # Use the valid index mapping
        actual_idx = self.valid_indices[idx]
        img_name = self.img_labels.iloc[actual_idx, 0]
        img_path = os.path.join(self.img_dir, img_name)
        label = self.img_labels.iloc[actual_idx, 1]

        try:
            image = Image.open(img_path).convert('RGB') # Ensure RGB
        except Exception as e:
            print(f"Error loading image {img_name}: {e}")
            # Fallback or handle error - maybe return a dummy or skip
            # For now, let's raise to see where it fails
            raise e

        if self.transform:
            image = self.transform(image)
        return image, label

# Example usage (simplified)
# train_dataset = CoffeeBeanDataset(img_dir='train_images', annotations_file='train_labels.csv', transform=train_transform)
# train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
```
The training process for the ResNet50, even with a relatively small dataset, took a good chunk of time on my laptop's GPU – maybe 5-6 hours per significant training run with different hyperparameters. After several iterations of tweaking learning rates and augmentation strategies, I managed to reach a performance metric I was satisfied with: **92% accuracy** on a held-out test set. This accuracy proved effective in distinguishing beans with clear defects from healthy ones, which was the primary goal. While not perfect, 92% felt like a solid outcome for a first attempt with a self-curated dataset.

To make the model accessible and demonstrate its capability, I built a proof-of-concept web interface using **Streamlit**. This part was relatively straightforward compared to the model training, taking about 2 days. Streamlit’s simple component-based structure made it easy to set up file uploads and display predictions. The main challenge was correctly loading the trained PyTorch model and running inference efficiently within the Streamlit app’s flow. I had to ensure the model was loaded only once and cached, as reloading it for every prediction request would have been prohibitively slow.

```python
# Snippet from the Streamlit app showing model loading and prediction
import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import os

@st.cache_resource # Cache the model to avoid reloading on every interaction
def load_model(model_path):
    # Define the model architecture (needs to match the trained model)
    # In a real app, this should perhaps be imported from your model script
    model = models.resnet50(weights=None) # Use weights=None as we load custom state_dict
    num_ftrs = model.fc.in_features
    num_classes = 2 # Assuming binary classification: Good/Defect
    model.fc = torch.nn.Linear(num_ftrs, num_classes)

    # Load the trained state dictionary
    try:
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'))) # Map to CPU for demo
        model.eval() # Set model to evaluation mode
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Define the image transformations (must match training pre-processing)
inference_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# --- Streamlit App ---
st.title("Coffee Bean Quality Classifier")

uploaded_file = st.file_uploader("Upload an image of a coffee bean...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Uploaded Image.", use_column_width=True)

    # Load the model
    # Need to specify the path to your saved model file
    model_path = "path/to/your/trained_model.pth" # <<< REPLACE with your actual path
    model = load_model(model_path)

    if model:
        # Preprocess the image
        input_tensor = inference_transform(image)
        input_batch = input_tensor.unsqueeze(0) # Add a batch dimension

        # Perform inference
        with torch.no_grad(): # Disable gradient calculation for inference
            output = model(input_batch)
            probabilities = torch.softmax(output, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = torch.max(probabilities).item() * 100

        # Define class names (replace with your actual class names)
        class_names = ["Good Bean", "Defect Bean"]
        prediction = class_names[predicted_class]

        st.subheader("Prediction:")
        st.write(f"The model predicts this is a **{prediction}** with **{confidence:.2f}%** confidence.")

```
Finally, I spent some time exploring model interpretability techniques. It wasn't enough to just get a prediction; I wanted to understand *why* the model classified a bean as defective. I looked into techniques like Gradient-weighted Class Activation Mapping (Grad-CAM) to visualize which parts of the image the CNN was focusing on. Implementing Grad-CAM on my PyTorch model took another day, involving hooking into the model's layers and calculating gradients. This proved insightful, often showing that the model correctly focused on visible flaws like cracks or discoloration spots, reinforcing confidence in its predictions for expected defect types. However, for some ambiguous cases, the attention maps were less clear, highlighting the model's limitations on novel or subtle defects not strongly represented in the training data.

Overall, this project was a significant learning experience, from managing a custom dataset and navigating PyTorch's training complexities to deploying a simple interface and peering into the model's decision-making process. It reinforced the practical challenges involved in building a real-world computer vision system, extending far beyond just training a model.