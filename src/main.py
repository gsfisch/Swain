from swain import Swain
from encoder import SwinEncoder
from decoder import SwinDecoder
import torch
from torchvision.utils import save_image
from PIL import Image
from transform import PIL_image_to_Torch_tensor


# Drive code for image stylization (model inference)
def main():
    # Create the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = SwinEncoder()
    decoder = SwinDecoder()
    model = Swain(encoder, decoder).to(device)
    print(device)


    # Simulate user input (two images)
    content = torch.randn(1, 3, 224, 224).to(device)
    style = torch.randn(1, 3, 224, 224).to(device)
    
    content = Image.open('../content_input/Change_of_Hearts.jpg')
    style = Image.open('../style_input/cubismo.jpg')

    content = PIL_image_to_Torch_tensor(content).unsqueeze(0).to(device)
    style = PIL_image_to_Torch_tensor(style).unsqueeze(0).to(device)

    print(content.shape)
    print(style.shape)

    # Model inference
    stylized_image = model(content, style)


    # Save inference
    save_image(stylized_image, "../output/inference.jpg")


if __name__ == "__main__":
    main()
