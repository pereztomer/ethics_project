import torch
import torchvision.transforms as transforms
from efficientnet_pytorch import EfficientNet

# Instantiate an EfficientNet model
model = EfficientNet.from_pretrained('efficientnet-b0')

# Define the normalization parameters for EfficientNet
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

# Define a transformation pipeline that includes normalization
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize])

# Load an image and apply the transformation pipeline
img = Image.open('/path/to/image.jpg')
img_tensor = transform(img)

# Add an extra dimension to the tensor to represent a batch of size 1
img_tensor = img_tensor.unsqueeze(0)

# Pass the tensor through the model and get the predictions
with torch.no_grad():
    logits = model(img_tensor)

# Perform a softmax activation to get the predicted class probabilities
probs = torch.softmax(logits, dim=1)

# Print the top 5 predicted classes and their probabilities
top5_probs, top5_indices = torch.topk(probs, k=5)
for i in range(5):
    class_index = top5_indices[0][i]
    class_prob = top5_probs[0][i]
    class_name = devkit.idx_to_class[class_index]
    print(f"Class {class_index}: {class_name} (probability {class_prob:.2f})")
