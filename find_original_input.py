import torch
import torch.nn.functional as F
import torchvision.models as models
from PIL import Image
import numpy as np
import torchvision.datasets as datasets
import torchvision


def train():
    pil_im = Image.open("Bernese-Mountain-Dog-On-White-01.jpg")

    # im_normalizer = torchvision.models.EfficientNet_B7_Weights.IMAGENET1K_V1.transforms()
    # # Load a pre-trained model
    # model = models.efficientnet_b7(pretrained=True)
    # # that model variables can not be updated!
    # model.eval()
    #
    # im_tensor = torch.unsqueeze(im_normalizer(pil_im), 0)
    # class_number = np.argmax(model(im_tensor).detach().numpy())

    model = models.efficientnet_b0(pretrained=True)

    # Desired output
    target_class = 239

    # Initialize the input variable - this is a variable - not a scalar!
    input_size = (3, 224, 224)
    input_var = torch.randn(1, *input_size, requires_grad=True)

    # Set up the optimizer
    lr = 0.1
    steps = 1000
    opt = torch.optim.SGD([input_var], lr=lr)

    # Perform backpropagation
    for i in range(steps):
        opt.zero_grad()

        # Forward pass
        logits = model(input_var)
        loss = F.cross_entropy(logits, torch.tensor([target_class]))

        # Backward pass
        loss.backward()
        opt.step()

        # Print progress
        if i % 100 == 0:
            print('Step {}: Loss = {}'.format(i, loss.item()))

        # Clamp the input variable to a valid range
        input_var.data = torch.clamp(input_var.data, -1, 1)

    # Convert the input variable to a PIL image
    input_np = input_var.detach().numpy()[0]
    input_np = np.transpose(input_np, (1, 2, 0))
    input_np = (input_np + 1) / 2 * 255
    input_np = np.clip(input_np, 0, 255).astype(np.uint8)
    input_img = Image.fromarray(input_np)

    # Save the image
    input_img.save('targeted_image.png')


if __name__ == '__main__':
    train()
