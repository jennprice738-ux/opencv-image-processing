# Real-Time Camera Acquisition and Color-Space Image Processing Using OpenCV

**MTRE 4820 – Machine Learning for Robot Perception, Project #1**
Jenn Price

A Python program that records live video from a laptop camera and runs a series of basic image-processing operations on one frame using OpenCV. The full write-up is in IEEE format: [Project1_IEEE_Paper.pdf](Project1_IEEE_Paper.pdf).

> The example figures in the paper were generated from a synthetic test video to show each processing step.

## What it does

1. Grabs **4,500 frames** from the internal camera and saves them to `live_camera.avi` (MJPG codec).
2. Reopens the video file and replays it.
3. For the **1,000th frame**:
   - Displays the frame and prints its shape
   - Shows the red, green and blue channels in three separate windows
   - Resizes the frame to **70%** of its original size
   - Converts it from RGB (BGR) to **HSV**
   - Sets the **V** (brightness) channel of every pixel to 50%
   - Plots the histogram of the **S** (saturation) channel
   - Applies **histogram equalization** to the S channel and plots the new histogram
   - Converts it back to RGB
   - Applies a **7×7 Gaussian filter** to remove noise
   - Displays the result and saves it as `modified_img.jpg`

## Running it

Requires Python 3 and a webcam.

```bash
pip install -r requirements.txt
python image_operation.py
```

Recording 4,500 frames takes about 3 minutes at 25 fps. When the replay finishes, all result windows stay open; press any key in an image window (and close the histogram windows) to exit.

## Output files

| File | Contents |
|---|---|
| `live_camera.avi` | The 4,500 recorded frames |
| `modified_img.jpg` | The final processed and filtered image |
| `hist_s.png` | Histogram of the S channel |
| `hist_s_equalized.png` | Histogram of the S channel after equalization |

These are generated when you run the program and are not stored in the repository.

## Built with

[OpenCV](https://opencv.org/) · [NumPy](https://numpy.org/) · [Matplotlib](https://matplotlib.org/)
