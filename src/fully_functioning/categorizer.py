import os
import numpy as np
from PIL import Image
from db_utils import loadFromDir, save_np_array, load_np_array, save_array, load_array
from db_validation import validateDB
from translator_network import EncoderDecoderNetwork
from repo.src.fully_functioning.translator_trainer import main as trainer
from repo.src.fully_functioning.translator_trainer import display_image_normalized as save_image

# Image categorization is not yet implemented
# Image input MUST be 45x45
# For the best results, line weight should be 1px

def main():
    train_network = False
    input_image_address = ""
    output_image_path = ""
    output_image_name = "test.jpg"
    saved_model = "encoder_decoder2.npz"
    
    if train_network:
        trainer()
        
    network = EncoderDecoderNetwork.load(saved_model)
    
    while (True):
        
        print("would you like to generate an image?")
        command = input()
        if (command == "no" or command == exit):
            exit()
        
        input_image_address, output_image_path = grab_addresses()
        
        print("The network is processing the image")
        
        if (input_image_address):
            
            input_image = Image.open(input_image_address).convert("L")
            input_image_array = np.asarray(input_image).flatten()
            output_image = network.forward(input_image_array)
            
            save_image(output_image, 45, 45, output_image_path + "/" + output_image_name)
            input_image_address = ""
            output_image_path = ""
            
def grab_addresses():
    
        print("Enter the path for a 45x45 input image file")
        input_image_address = input()
        print("Enter a path for the output image")
        output_image_address = input()
        
        if not os.path.exists(input_image_address):
            print("invalid input address")
            
        if not os.path.exists(output_image_address):
            print("invalid output address")
        
        if os.path.exists(input_image_address) and os.path.exists(output_image_address):
            return input_image_address, output_image_address
        
        else:
            return False, False

if __name__ == "__main__":
    main()
