import torchvision.transforms as transforms


# Transform that convert PIL image to PyTorch tensor
def PIL_image_to_Torch_tensor():
    transform = transforms.Compose([transforms.Resize((224, 224)),
                                    transforms.ToTensor()
                                    ])
    
    return transform


# Transform that convert PyTorch tensor to PIL image 
def Torch_tensor_to_PIL_image():
    transform = transforms.ToPILImage()

    return transform