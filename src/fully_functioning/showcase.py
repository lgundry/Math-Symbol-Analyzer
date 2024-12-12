
from tkinter import *
from tkinter import filedialog
from PIL import Image, ImageTk
from full_network import EncoderDecoderNetwork
import numpy as np
from db_utils import load_array
import os

# File explorer portion of code copied from https://www.geeksforgeeks.org/file-explorer-in-python-using-tkinter/
class Showcase:
    def __init__(self):
        window = Tk()
        window.title('Neural Network Showcase')
        window.geometry("900x500")
        window.config(background = "white")
        self.label_file_explorer = Label(window, text = "Select a file to give to network", width = 100, height = 4, fg = "blue")
        
        self.network = EncoderDecoderNetwork("encoder_decoder3.npz")
        self.labels = load_array("labels.pkl")
        
        # File explorer stuff
        self.button_explore = Button(window, text = "Browse Files", command = self.browseFiles) 
        self.button_exit = Button(window, text = "Exit", command = exit)
        
        # Network frame stuff
        self.network_frame = Frame(window, width=100, height=5, bg="white")
        self.input_image_label = Label(self.network_frame, image = "", text="", bg="white")
        self.output_image_label = Label(self.network_frame, image = "", text="", bg="white")
        self.actual_symbol_label = Label(self.network_frame, image = "", text="", bg="white")
        self.output_label_description_label = Label(self.network_frame, text = "", anchor="w", justify="left", wraplength=500, bg="white")
        
        # Start Button
        self.button_start = Button(window, text="Begin", command = self.network_forward_pass)
        
        # Put pieces onto the window
        self.label_file_explorer.grid(column = 0, row = 0)
        self.button_explore.grid(column = 0, row = 1)
        self.button_start.grid(column=0, row = 4)
        self.button_exit.grid(column = 0,row = 5)
        
        self.network_frame.grid(column = 0, row = 3)
        self.input_image_label.grid(column = 0, row = 0)
        self.output_image_label.grid(column = 1, row = 0)
        self.actual_symbol_label.grid(column = 2, row = 0)
        self.output_label_description_label.grid(column=4, row=0)
        
        # Empty rows
        self.spacer = Label(window, text="", height=2, bg="white")
        self.spacer.grid(column=0, row=2)  # Place before the network frame
        
        self.spacer = Label(window, text="", height=2, bg="white")
        self.spacer.grid(column=0, row=2)  # Place before the network frame
        
        # Let the window wait for any events
        window.mainloop()

    def browseFiles(self):
        self.filename = filedialog.askopenfilename(initialdir = "~", title = "Select a File", filetypes = (("Image files", "*.jpg*"), ("all files", "*.*")))
        # Change label contents
        self.label_file_explorer.configure(text=f"Selected {self.filename} for processing")
        self.open_image()
        
    def open_image(self):
        self.input_image = Image.open(self.filename).convert("L")
        input_image = self.input_image.resize(((90, 90)), Image.Resampling.LANCZOS)
        self.photo_input = ImageTk.PhotoImage(input_image)
        self.input_image_label.configure(image = self.photo_input, text="Image given", bg="gray")
        
    def network_forward_pass(self):
        network_input = np.asarray(self.input_image).flatten()
        
        output_image, label_array = self.network.forward(network_input)
        output_image = output_image.reshape((45, 45))
        output_image = np.clip(output_image * 255, 0, 255).astype(np.uint8)
        output_image = Image.fromarray(output_image)
        output_image = output_image.resize(((90, 90)), Image.Resampling.LANCZOS)
        self.photo_output = ImageTk.PhotoImage(output_image)
        
        self.label = self.labels[np.argmax(label_array)]
        
        with open(f"definitions/{self.label}.txt", "r") as file:
            self.label_description = file.read().strip()
            
        actual_symbol = Image.open(f"definitions/{self.label}.jpg")
        actual_symbol = actual_symbol.resize(((90, 90)), Image.Resampling.LANCZOS)
        self.photo_actual_symbol = ImageTk.PhotoImage(actual_symbol)
        
        self.output_image_label.configure(image=self.photo_output, text="Image generated", bg="gray")
        self.actual_symbol_label.configure(image = self.photo_actual_symbol, text="Symbol image (based off categorization)", bg="gray")
        self.output_label_description_label.configure(text=self.label_description, bg="gray")
        self.network_frame.configure(bg="gray")
        
        
def main():
    a_showcase = Showcase()
    
if __name__ == "__main__":
    main()
