from encoder import SwinEncoder
from decoder import SwinDecoder
import torch
import torch.nn as nn
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torchvision import datasets
from transform import PIL_image_to_Torch_tensor, Torch_tensor_to_PIL_image
import torch.optim as optim
import os


# Drive code for Swin Transformer decoder training 
def main():
    # Parameters
    batch_size = 8
    learning_rate = 1e-5


    # Set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)


    # Image Transform
    transform = PIL_image_to_Torch_tensor()


    # DataLoader
    dataset = datasets.ImageFolder("../dataset/style/", transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


    # Load models
    encoder = SwinEncoder().to(device)
    decoder = SwinDecoder().to(device)


    model_dir = "../models"
    input_model_file = "decoder_Stage1_40000_steps.pth"
    output_model_file = "decoder_Stage1_50000_steps.pth"
    os.makedirs(model_dir, exist_ok=True)
    input_checkpoint_path = model_dir + "/" + input_model_file
    output_checkpoint_path = model_dir + "/" + output_model_file


    # Load Decoder if there is one
    if os.path.exists(input_checkpoint_path):
        decoder.load_state_dict(torch.load(input_checkpoint_path, map_location=device))
        print("Loading checkpoint.")
    else:
        print("Starting training from scratch.")


    # Freeze encoder
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False


    # Set optimizer
    optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)
    loss_fn = nn.L1Loss()


    # Training
    decoder.train()

    for step, (imgs, _) in enumerate(loader):
        imgs = imgs.to(torch.float32).to(device)

        with torch.no_grad():
            features = encoder(imgs) # List of 4 feature maps

        output = decoder(features[0].to(device))

        loss = loss_fn(output, imgs)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Step {step} - L1 Loss: {loss.item():.4f}")

        if step == 9999:
            torch.save(decoder.state_dict(), output_checkpoint_path)
            print("Training complete. Decoder saved.")  
            break

    return


if __name__ == "__main__":
    main()