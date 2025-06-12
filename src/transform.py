import torchvision.transforms as transforms


# Convert image from PIL to PyTorch tensor
def PIL_image_to_Torch_tensor(image):
    # Convert the PIL image to Torch tensor

    transform = transforms.Compose([transforms.PILToTensor()])
    
    image_tensor = transform(image)

    return image_tensor