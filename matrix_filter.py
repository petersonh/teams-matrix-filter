# region Imports

import pyvirtualcam
import numpy as np
import cv2
import random
from PIL import Image, ImageDraw, ImageFont
import os

# endregion


# region Functions

def clamp(n, smallest, largest):
    return max(smallest, min(n, largest))

def render(img_rgb):
    numDownSamples = 2 # number of downscaling steps
    numBilateralFilters = 7  # number of bilateral filtering steps

    # -- STEP 1 --
    # downsample image using Gaussian pyramid
    img_color = img_rgb
    for _ in range(numDownSamples):
        img_color = cv2.pyrDown(img_color)

    # repeatedly apply small bilateral filter instead of applying
    # one large filter
    for _ in range(numBilateralFilters):
        img_color = cv2.bilateralFilter(img_color, 9, 9, 7)

    # upsample image to original size
    for _ in range(numDownSamples):
        img_color = cv2.pyrUp(img_color)

    # -- STEPS 2 and 3 --
    # convert to grayscale and apply median blur
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    img_blur = cv2.medianBlur(img_gray, 7)

    # -- STEP 4 --
    # detect and enhance edges
    img_edge = cv2.adaptiveThreshold(img_blur, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 3, 2)
    
#     return img_edge

    # -- STEP 5 --
    # convert back to color so that it can be bit-ANDed
    # with color image
    img_edge = cv2.cvtColor(img_edge, cv2.COLOR_GRAY2RGB)
#     return img_edge
    return cv2.bitwise_and(img_color, img_edge)


# endregion


# region Type Definitions

class MatrixLine:
    def __init__(self, size=20, trail_length=10, height=360, x=0):
        
        self.size = size
        self.trail_length = trail_length
        self.height = height
        self.x = x
        self.characters = self.reset_characters()
        self.max_alpha_index = random.randint(0, size - 1)
        self.alpha_increment = 1 / trail_length
        self.wait = 0
        self.set_alphas()
    
    def reset_characters(self):
        return [{"character": random.choice(list("いろはにほへとちりぬるをわかよたれ")), "alpha": 0.0, "y": (x / self.size) * self.height} for x in range(self.size)]
    
    def set_alphas(self):
        current_index = self.max_alpha_index
        current_alpha = 1.0
        while current_index >= 0:
            if current_index < self.size:
                self.characters[current_index]["alpha"] = clamp(current_alpha, 0.0, 1.0)
            current_index -= 1
            current_alpha -= self.alpha_increment
            
    def cycle_alphas(self):
        if self.wait == 0:
            self.max_alpha_index += 1
            if self.max_alpha_index - self.trail_length == self.size:
                self.reset_characters()
                self.max_alpha_index = 0
                self.wait = random.randint(0,10)
            self.set_alphas()
        else:
            self.wait -= 1

# endregion


# region Constants

WIDTH = 640
HEIGHT = 360
FPS = 30
MINUTES_TO_CAPTURE = 1
FRAMES_TO_CAPTURE = int(FPS * 60 * MINUTES_TO_CAPTURE)

# endregion


# region Main Program

# naively checking for the presence of the background frames and generating if needed
if os.path.isdir("matrix_frames") is False:

    print("Generating background frames...")
    os.mkdir("matrix_frames")
    matrix_lines_left = []
    number_of_columns = 20
    padding = 10
    for column in range(number_of_columns):
        x = int((column / number_of_columns) * (WIDTH / 3)) + padding
        matrix_lines_left.append(MatrixLine(20, x=x))

    matrix_lines_right = []
    padding = (WIDTH / 3) * 2
    for column in range(number_of_columns):
        x = int((column / number_of_columns) * (WIDTH / 3)) + padding
        matrix_lines_right.append(MatrixLine(20, x=x))

    matrix_lines = matrix_lines_left + matrix_lines_right

    for count in range(FRAMES_TO_CAPTURE):
        txt = Image.new('RGBA', (WIDTH, HEIGHT), (255,255,255,0))
        font = ImageFont.truetype("Gen Jyuu Gothic Monospace Bold.ttf", 10)
        d = ImageDraw.Draw(txt)
        for matrix_line in matrix_lines:
            for character in matrix_line.characters:
                alpha = int(character['alpha'] * 255)
                y = character['y']
                x = matrix_line.x
                c = character['character']
                d.text((x, y), c, fill=(255, 255, 255, alpha), font=font)
            matrix_line.cycle_alphas()

        txt.save("matrix_frames/frame_" + str(count) + ".png")
        percent_complete = str(round((count / FRAMES_TO_CAPTURE) * 100, 2))
        print(percent_complete + "%")
    
    print("Done!")

matrix_lines = []
number_of_columns = 40
padding = 10
for column in range(number_of_columns):
    x = int((column / number_of_columns) * 640) + padding
    matrix_lines.append(MatrixLine(20, x=x))

print("Starting virtual camera, press q to quit...")
with pyvirtualcam.Camera(width=WIDTH, height=HEIGHT, fps=30) as cam:

    vid = cv2.VideoCapture(0)
    vid.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    vid.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    start = vid.get(cv2.CAP_PROP_POS_FRAMES)
    
    count = 0
    while(True):

        ret, frame = vid.read()
        
        # main effects are in this render functions
        frame = render(frame)
        # flipping the bits here for that inverted color look
        frame = cv2.bitwise_not(frame)
        
        image = Image.fromarray(frame).convert("RGBA")         
        matrix_image = Image.open("matrix_frames/frame_" + str(count) + ".png")
        image = Image.alpha_composite(image, matrix_image).convert("RGB")
        frame = np.array(image)
        frame = cv2.applyColorMap(frame, cv2.COLORMAP_DEEPGREEN)
        cv2.imshow('frame', frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cam.send(frame)
        cam.sleep_until_next_frame()

        count += 1
        if count == FRAMES_TO_CAPTURE:
            count = 0
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # After the loop release the cap object
    vid.release()
    # Destroy all the windows
    cv2.destroyAllWindows()

# endregion