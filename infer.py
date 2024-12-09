import torch
from sketch_rnn.model import SketchRNN, sample_unconditional, sample_conditional
from sketch_rnn.hparams import hparams
from sketch_rnn.dataset import load_strokes, SketchRNNDataset

import matplotlib.pyplot as plt
import numpy as np
import drawsvg as draw
import PIL
from PIL import Image
import svgwrite
import os

def load_npz_to_tensor(npz_file):
    """
    Loads sketch data from a .npz file into PyTorch tensors.
    Each sketch is handled as a list of tensors due to variable lengths.
    """
    data = np.load(npz_file, encoding='latin1', allow_pickle=True)
    train_strokes = data['train']
    valid_strokes = data['valid']
    test_strokes = data['test']
    
    # Convert each sketch into a tensor
    train_strokes = [torch.tensor(stroke, dtype=torch.float32) for stroke in train_strokes]
    valid_strokes = [torch.tensor(stroke, dtype=torch.float32) for stroke in valid_strokes]
    test_strokes = [torch.tensor(stroke, dtype=torch.float32) for stroke in test_strokes]

    return train_strokes, valid_strokes, test_strokes


def get_bounds(data, factor):
    """
    Calculate the bounding box for the sketch.
    """
    x_cumsum = np.cumsum(data[:, 0]) / factor
    y_cumsum = np.cumsum(data[:, 1]) / factor
    min_x, max_x = np.min(x_cumsum), np.max(x_cumsum)
    min_y, max_y = np.min(y_cumsum), np.max(y_cumsum)
    return min_x, max_x, min_y, max_y

def draw_strokes(data, factor=0.2, svg_filename='sample.svg', show=False):
    """
    Draws the strokes in SVG format, saves the file, and optionally displays it.
    
    Args:
    - data (torch.Tensor): Tensor of stroke data with shape (N, 3).
    - factor (float): Scaling factor for the sketch.
    - svg_filename (str): Path to save the SVG file.
    - show (bool): If True, opens the SVG file in the default viewer.
    """
    data = data.numpy()  # Convert PyTorch tensor to NumPy array if needed
    min_x, max_x, min_y, max_y = get_bounds(data, factor)
    dims = (50 + max_x - min_x, 50 + max_y - min_y)
    dwg = svgwrite.Drawing(svg_filename, size=dims)
    dwg.add(dwg.rect(insert=(0, 0), size=dims, fill='white'))
    
    lift_pen = 1
    abs_x = 25 - min_x
    abs_y = 25 - min_y
    p = "M%s,%s " % (abs_x, abs_y)
    command = "m"
    
    for i in range(len(data)):
        if lift_pen == 1:
            command = "m"
        elif command != "l":
            command = "l"
        else:
            command = ""
        x = float(data[i, 0]) / factor
        y = float(data[i, 1]) / factor
        lift_pen = data[i, 2]
        p += command + str(x) + "," + str(y) + " "
    
    the_color = "black"
    stroke_width = 1
    dwg.add(dwg.path(p).stroke(the_color, stroke_width).fill("none"))
    dwg.save()

    print(f"SVG saved as {svg_filename}")
    if show:
        os.system(f"start {svg_filename}" if os.name == 'nt' else f"open {svg_filename}")

def generate():

    # Assume `hps` contains hyperparameters used during training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the trained model
    hps = hparams()
    model = SketchRNN(hps)
    model.load_state_dict(torch.load('profiles/200epochs/model_save/model.pt', map_location=device))

    # load sample
    train_strokes, valid_strokes, test_strokes = load_npz_to_tensor('data/cat.npz')
    cat_sample_data = test_strokes[0]

    draw_strokes(cat_sample_data, factor=0.8, svg_filename="true_cat.svg")
    print(cat_sample_data)
    # conditional sampling
    sampled_s, sampled_p = sample_unconditional(model, T=1, device=device)
    sampled_strokes = []
    for i in range(sampled_s.shape[0]):
        dx, dy = sampled_s[i].tolist()
        pen_state = sampled_p[i].item()
        sampled_strokes.append([dx, dy, pen_state])
    draw_strokes(torch.tensor(sampled_strokes, dtype=torch.float32), factor=0.02, svg_filename="fake_cat.svg")

    print("Sampled X (strokes):", sampled_s)
    print("Sampled V (pen state):", sampled_p)

