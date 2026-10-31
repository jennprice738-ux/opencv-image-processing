### Jenn Price
# MTRE 4820 - Machine Learning for Robot Perception - Project #1
# Basic image processing with OpenCV: record, replay, and process the 1000th frame.

import sys

import cv2
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Settings (all the "magic numbers" from the project handout live here)
# ---------------------------------------------------------------------------
CAMERA_INDEX = 0                 # 0 = laptop's internal (default) camera
TOTAL_FRAMES = 4500              # task 1: grab 4,500 frames
VIDEO_FILE = "live_camera.avi"   # task 2: output video file
OUTPUT_IMAGE = "modified_img.jpg"
TARGET_FRAME = 1000              # task 5: process the 1,000th frame
DEFAULT_FPS = 25.0               # used if the camera does not report its fps
RESIZE_SCALE = 0.70              # resize to 70% of the original size
V_VALUE = 0.50                   # assign 50% to V (0.50 * 255 = 128)
GAUSS_KERNEL = (7, 7)            # 7 x 7 Gaussian filter


def record_video():
    """Tasks 1-2: grab TOTAL_FRAMES frames from the camera and save them to VIDEO_FILE."""
    capture = cv2.VideoCapture(CAMERA_INDEX)
    if not capture.isOpened():
        sys.exit("Error: could not open the camera.")

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = capture.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 1 or fps > 120:      # many webcams report 0
        fps = DEFAULT_FPS
    print(f"Camera opened: {width} x {height} at {fps:.1f} fps")

    # MJPG codec in an AVI container; frame size must match the camera frames exactly
    writer = cv2.VideoWriter(VIDEO_FILE, cv2.VideoWriter_fourcc(*"MJPG"), fps, (width, height))
    if not writer.isOpened():
        capture.release()
        sys.exit(f"Error: could not create {VIDEO_FILE}.")

    count = 0
    while count < TOTAL_FRAMES:
        ok, frame = capture.read()
        if not ok:
            print("Warning: camera stopped returning frames.")
            break
        writer.write(frame)
        count += 1

        cv2.imshow("Live Camera", frame)
        cv2.waitKey(1)                          # lets the OpenCV window refresh
        if count % 500 == 0:                    # progress update (printing every frame is slow)
            print(f"Recorded {count}/{TOTAL_FRAMES} frames")

    capture.release()
    writer.release()
    cv2.destroyAllWindows()
    print(f"Saved {count} frames to {VIDEO_FILE}")
    return fps


def show_channels(frame):
    """Show the blue, green and red channels in three separate windows."""
    blue, green, red = cv2.split(frame)        # OpenCV stores colour images as B, G, R
    zeros = np.zeros_like(blue)
    # Each channel is shown in its own colour (the other two channels set to 0)
    cv2.imshow("Red Channel", cv2.merge([zeros, zeros, red]))
    cv2.imshow("Green Channel", cv2.merge([zeros, green, zeros]))
    cv2.imshow("Blue Channel", cv2.merge([blue, zeros, zeros]))


def plot_histogram(channel, title, filename):
    """Plot a 256-bin histogram of an 8-bit channel in its own window and save it."""
    hist = cv2.calcHist([channel], [0], None, [256], [0, 256]).ravel()
    plt.figure(title)
    plt.bar(np.arange(256), hist, width=1.0)
    plt.xlim(0, 255)
    plt.xlabel("S value (0-255)")
    plt.ylabel("Number of pixels")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.show(block=False)
    plt.pause(0.001)                           # draw the figure without stopping the program


def process_frame(frame):
    """Task 5: every operation required on the 1,000th frame."""
    # Show the frame and print its shape
    cv2.imshow(f"Frame {TARGET_FRAME}", frame)
    print(f"Shape of frame {TARGET_FRAME} (rows, cols, channels): {frame.shape}")

    # Red, green and blue channels in three extra windows
    show_channels(frame)

    # Resize to 70% of the original size (INTER_AREA is the best choice for shrinking)
    resized = cv2.resize(frame, None, fx=RESIZE_SCALE, fy=RESIZE_SCALE,
                         interpolation=cv2.INTER_AREA)
    print(f"Shape after resizing to 70%: {resized.shape}")

    # Colour space conversion: BGR (OpenCV's RGB ordering) -> HSV
    hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)

    # Assign 50% to the V (value/brightness) channel of every pixel
    hsv[:, :, 2] = np.uint8(round(V_VALUE * 255))

    # Histogram of the S (saturation) channel
    s_channel = hsv[:, :, 1]
    plot_histogram(s_channel, "Histogram of S Channel", "hist_s.png")

    # Equalize the S channel and show its new histogram
    s_equalized = cv2.equalizeHist(s_channel)
    hsv[:, :, 1] = s_equalized
    plot_histogram(s_equalized, "Equalized Histogram of S Channel", "hist_s_equalized.png")

    # Convert the processed image back: HSV -> BGR (RGB)
    processed = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # 7 x 7 Gaussian filter to remove noise (sigma computed from the kernel size)
    filtered = cv2.GaussianBlur(processed, GAUSS_KERNEL, 0)
    cv2.imshow("Filtered Image", filtered)

    # Save the filtered image and confirm it was written
    if cv2.imwrite(OUTPUT_IMAGE, filtered):
        print(f"Saved {OUTPUT_IMAGE} with shape {filtered.shape}")
    else:
        print(f"Error: could not save {OUTPUT_IMAGE}")


def replay_video(fps):
    """Tasks 3-5: open the saved video, replay it, and process the 1,000th frame."""
    capture = cv2.VideoCapture(VIDEO_FILE)
    if not capture.isOpened():
        sys.exit(f"Error: could not open {VIDEO_FILE}.")
    print(f"{VIDEO_FILE} contains {int(capture.get(cv2.CAP_PROP_FRAME_COUNT))} frames")

    delay_ms = max(1, int(1000 / fps))         # replay at the recorded speed
    frame_number = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            print("End of video.")
            break
        frame_number += 1
        cv2.imshow("Video Replay", frame)

        if frame_number == TARGET_FRAME:
            process_frame(frame)

        if cv2.waitKey(delay_ms) & 0xFF == 27:  # Esc stops the replay early
            print("Replay stopped by user.")
            break

    capture.release()
    if frame_number < TARGET_FRAME:
        print(f"Warning: the video only has {frame_number} frames, "
              f"so frame {TARGET_FRAME} was never reached.")


def main():
    fps = record_video()
    replay_video(fps)

    # Keep every result window open so they can be inspected
    print("Done. Close the histogram windows (or press any key in an image window) to exit.")
    cv2.waitKey(0)
    plt.show()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
