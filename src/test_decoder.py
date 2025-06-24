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

    # Image Transforms
    transform_input = PIL_image_to_Torch_tensor()
    transform_output = Torch_tensor_to_PIL_image()

    # DataLoader
    dataset = datasets.ImageFolder("../dataset/style/", transform=transform_input)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    encoder = SwinEncoder().to(device)
    decoder = SwinDecoder().to(device)

    # Load trained decoder
    checkpoint_path = '../models/decoder_Stage1_50000_steps.pth'
    decoder.load_state_dict(torch.load(checkpoint_path, map_location=device))

    # Freeze encoder
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False

    # Freeze decoder
    decoder.eval()
    for p in decoder.parameters():
        p.requires_grad = False


    # Inference
    for step, (img, _) in enumerate(loader):
        img = img.to(device)

        # Tensor to image
        img_transformed = transform_output(img[0])

        with torch.no_grad():
            features = encoder(img) # List of 4 feature maps
            output = decoder(features[0].to(device))

        reconstructed_image = display_fix(output[0])

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