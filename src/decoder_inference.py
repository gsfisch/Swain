import torch
from encoder import SwinEncoder
from decoder import SwinDecoder
from PIL import Image
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.utils import save_image
from transform import PIL_image_to_Torch_tensor, Torch_tensor_to_PIL_image


def display_fix(image):
    save_image(image, "../output/temp.jpg")
    image = Image.open("../output/temp.jpg")

    return image


# Drive for Decoder test
def main():
    # Set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    # Param
    batch_size = 1
    image = Image.open('../dataset/style/Baroque/adriaen-van-de-velde_the-farm-1666.jpg')

    # Image Transforms
    transform_input = PIL_image_to_Torch_tensor()
    transform_output = Torch_tensor_to_PIL_image()


    encoder = SwinEncoder().to(device)
    decoder = SwinDecoder().to(device)

    # Load trained decoder
    checkpoint_path = '../models/decoder_Stage4_20000_steps.pth'
    decoder.load_state_dict(torch.load(checkpoint_path, map_location=device))

    # Freeze encoder
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False

    # Freeze decoder
    decoder.eval()
    for p in decoder.parameters():
        p.requires_grad = False


    # Tensor to image
    image = transform_input(image)

    # Inference
    image = image.to(device)

    with torch.no_grad():
        features = encoder(image.unsqueeze(0)) # List of 4 feature maps
        output = decoder(features[3].to(device))

    reconstructed_image = display_fix(output[0])
    save_image(reconstructed_image, "../output/inference.jpg")

    # Show original and reconstructed images
    plt.subplot(1, 2, 1)
    plt.title('Original Image')
    plt.imshow(img_transformed)

    plt.subplot(1, 2, 2)
    plt.title('Reconstructed Image')
    plt.imshow(reconstructed_image)

    plt.show()


if __name__ == "__main__":
    main()