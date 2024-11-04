from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

jpg_filepath = "data/extracted_images/exists/exists_2225.jpg"

jpg_plt_img = Image.open(jpg_filepath)

jpg_as_np_array = np.asarray(jpg_plt_img)

np.save("outfile.txt", jpg_as_np_array)