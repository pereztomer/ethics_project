import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms


def train():
    # Check if GPU is available, otherwise use CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    # Load a pre-trained VGG16 model and move it to the device
    model = models.vgg16(pretrained=True)
    model.to(device)
    model.eval()

    # Define the target class (e.g. "dog")
    target_class = 263

    # Generate a random noise image as the starting point and move it to the device
    input_tensor = torch.randn((1, 3, 224, 224), requires_grad=True, device=device)

    # Set up the optimizer and criterion for backpropagation
    optimizer = torch.optim.Adam([input_tensor], lr=0.1)
    criterion = torch.nn.CrossEntropyLoss()

    # Perform backpropagation to generate a new image that maximizes the target class
    for i in range(1000):
        optimizer.zero_grad()
        output = model(input_tensor)
        loss = criterion(output, torch.tensor([target_class], device=device))
        loss.backward()
        optimizer.step()
        # Clamp the pixel values to the range [0, 1]
        input_tensor.data.clamp_(0, 1)

    # Convert the tensor back to an image and un-normalize it
    np_output = np.array(input_tensor.detach().cpu().squeeze())
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    output_image = transforms.functional.to_pil_image(input_tensor.cpu().squeeze())
    unnormalize = transforms.Normalize(mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
                                       std=[1 / 0.229, 1 / 0.224, 1 / 0.225])
    output_image = transforms.functional.to_pil_image(unnormalize(input_tensor.cpu().squeeze()))

    # Save or display the generated image
    output_image.save('generated_image.jpg')
    output_image.show()


if __name__ == '__main__':
    train()
