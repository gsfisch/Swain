from swain import Swain
from encoder import SwinEncoder
from decoder import SwinDecoder
import torch
from torchvision.utils import save_image
from torchvision import datasets
from torch.utils.data import DataLoader
import torch.optim as optim
from PIL import Image
from transform import PIL_image_to_Torch_tensor, Torch_tensor_to_PIL_image
import matplotlib.pyplot as plt
import os
import numpy as np


def display_fix(image):
    save_image(image, "../output/temp.jpg")
    image = Image.open("../output/temp.jpg")

    return image


def main():
    # Parameters
    batch_size = 8
    learning_rate = 1e-4
    model_dir = "../models"


    # Create model dir
    os.makedirs(model_dir, exist_ok=True)
    input_decoder_weights_path = model_dir + "/decoder_Stage1_50000_steps_style_transfer_1000_steps.pth" #"/decoder_Stage1_50000_steps.pth"
    output_decoder_weights_path = model_dir + "/decoder_Stage1_50000_steps_style_transfer_1000_steps.pth"


    # Set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)


    # Image Transforms
    transform_input = PIL_image_to_Torch_tensor()
    transform_output = Torch_tensor_to_PIL_image()


    # DataLoader
    dataset = datasets.ImageFolder("../dataset/style/", transform=transform_input)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


    # Load models
    encoder = SwinEncoder().to(device)
    decoder = SwinDecoder().to(device)

    # Load Decoder weights
    if os.path.exists(input_decoder_weights_path):
        decoder.load_state_dict(torch.load(input_decoder_weights_path, map_location=device))
        print("Loading decoder weights.")
    else:
        print("No decoder weights found.")
        return


    # Load model
    model = Swain(encoder, decoder).to(device)


    # Freeze encoder
    model.encoder.eval()
    for p in model.encoder.parameters():
        p.requires_grad = False    


    # Set optimizer
    optimizer = optim.Adam(model.decoder.parameters(), lr=learning_rate)

    # Training
    model.decoder.train()

    for step, (style_img, content_img) in enumerate(zip(loader, loader)):
        style_img, _ = style_img        # Remove label
        content_img, _ = content_img    # Remove label

        style_img = style_img.to(torch.float32).to(device)
        content_img = content_img.to(torch.float32).to(device)

        stylized_img, loss_c, loss_s = model(content_img, style_img)
        
        plt.subplot(1, 3, 1)
        plt.title('Content img')
        plt.imshow(transform_output(content_img[0]))

        plt.subplot(1, 3, 2)
        plt.title('Style img')
        plt.imshow(transform_output(style_img[0]))

        plt.subplot(1, 3, 3)
        plt.title('Stylized img')
        #plt.imshow(display_fix(transform_output(stylized_img[0])))
        plt.imshow(display_fix(stylized_img[0]))

        plt.show()
        continue
        

        loss = 0.0*loss_c + 1.0*loss_s

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Step {step} - L1 Loss: {loss.item():.4f}")

        if step == 999:
            torch.save(decoder.state_dict(), output_decoder_weights_path)
            print("Training complete. Decoder saved.")  
            break

        continue
        ''' 
        plt.subplot(1, 3, 1)
        plt.title('Content img')
        plt.imshow(transform_output(content_img[0]))

        plt.subplot(1, 3, 2)
        plt.title('Style img')
        plt.imshow(transform_output(style_img[0]))

        plt.subplot(1, 3, 3)
        plt.title('Stylized img')
        #plt.imshow(display_fix(transform_output(stylized_img[0])))
        plt.imshow(transform_output(display_fix(stylized_img)))

        plt.show()
        continue
        
        print('Style img:')
        print(style_img.shape)

        print('Content img:')
        print(content_img.shape)

        # Show original and reconstructed images
        plt.subplot(2, 8, 1)
        plt.title('style_img')
        plt.imshow(transform_output(style_img[0]))
        plt.subplot(2, 8, 2)
        plt.imshow(transform_output(style_img[1]))
        plt.subplot(2, 8, 3)
        plt.imshow(transform_output(style_img[2]))
        plt.subplot(2, 8, 4)
        plt.imshow(transform_output(style_img[3]))
        plt.subplot(2, 8, 5)
        plt.imshow(transform_output(style_img[4]))
        plt.subplot(2, 8, 6)
        plt.imshow(transform_output(style_img[5]))
        plt.subplot(2, 8, 7)
        plt.imshow(transform_output(style_img[6]))
        plt.subplot(2, 8, 8)
        plt.imshow(transform_output(style_img[7]))


        plt.subplot(2, 8, 9)
        plt.title('style_img')
        plt.imshow(transform_output(content_img[0]))
        plt.subplot(2, 8, 10)
        plt.imshow(transform_output(content_img[1]))
        plt.subplot(2, 8, 11)
        plt.imshow(transform_output(content_img[2]))
        plt.subplot(2, 8, 12)
        plt.imshow(transform_output(content_img[3]))
        plt.subplot(2, 8, 13)
        plt.imshow(transform_output(content_img[4]))
        plt.subplot(2, 8, 14)
        plt.imshow(transform_output(content_img[5]))
        plt.subplot(2, 8, 15)
        plt.imshow(transform_output(content_img[6]))
        plt.subplot(2, 8, 16)
        plt.imshow(transform_output(content_img[7]))

        plt.show()

        return
        '''

        with torch.no_grad():
            features = encoder(imgs) # List of 4 feature maps

        output = decoder(features[0].to(device))

        loss = loss_fn(output, imgs)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Step {step} - L1 Loss: {loss.item():.4f}")

        if step == 999:
            torch.save(decoder.state_dict(), output_decoder_weights_path)
            print("Training complete. Decoder saved.")  
            break

    return


''' 
# Drive code for image stylization (model inference)
def main():
    input_checkpoint_path = "../models/decoder_Stage1_50000_steps.pth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)


    # Create the model
    encoder = SwinEncoder()
    decoder = SwinDecoder()

    if os.path.exists(input_checkpoint_path):
        decoder.load_state_dict(torch.load(input_checkpoint_path, map_location=device))
        print("Loading checkpoint.")
    else:
        print("No decoder weights found.")
        return

    model = Swain(encoder, decoder).to(device)    


    # Network input
    content = Image.open('../content_input/Change_of_Hearts.jpg')
    style = Image.open('../style_input/cubismo.jpg')


    # Transform image to tensor and add Batch axis to it
    transform_input = PIL_image_to_Torch_tensor()
    transform_output = Torch_tensor_to_PIL_image()


    # Prepare input
    content = transform_input(content).unsqueeze(0).to(device)
    style = transform_input(style).unsqueeze(0).to(device)


    # Model inference
    model.eval()
    stylized_image, loss_c, loss_s = model(content, style)
    save_image(stylized_image, "../output/inference.jpg")
    #stylized_image = transform_output(stylized_image[0])
    #print('After inference')

    #print(type(stylized_image))

    stylized_image = Image.open("../output/inference.jpg")

    plt.imshow(stylized_image, cmap='gray')
    plt.show()
'''

if __name__ == "__main__":
    main()
