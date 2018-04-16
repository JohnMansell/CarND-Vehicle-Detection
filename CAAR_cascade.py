import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt
from scipy.ndimage.measurements import label

car_cascade = cv2.CascadeClassifier('cars.xml')

test_images = glob.glob('test_images/*.jpg')

heatmap_array = []

min_threshold = 10


'''-------------------------
        Draw Labeled Boxes
----------------------------'''
def draw_labeled_bboxes(img, labels, color=(0, 0, 255), thick=6):
    for car_number in range(1, labels[1] + 1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()

        # Identify (x,y) values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        # Define a bounding box based on min/ max of (x,y)
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], color, thick)

    # Return the img
    return img

'''-------------------------
        Apply Threshold
----------------------------'''
def apply_threshold(heatmap, threshold):
    heatmap[heatmap <= threshold] = 0
    return heatmap

'''-------------------------
        Show
----------------------------'''
def show(img, img2=None, img3=None, title=None):

    # 1 Image
    if (img2 is None):
        plt.imshow(img, cmap='gray')

    # 2 Images
    elif img3 is None:
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        ax1.imshow(img, cmap='gray')
        ax2.imshow(img2, cmap='gray')

    else:
        f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 10))
        ax1.imshow(img, cmap='gray')
        ax2.imshow(img2, cmap='gray')
        ax3.imshow(img3, cmap='gray')

    if(title is not None):
        plt.savefig(title, bbox_inches='tight')

    plt.show()


'''============================
        Process Frame
==============================='''
def process_frame(img):
    draw_img = np.copy(img)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    heatmap = np.zeros_like(gray)
    car_array = []

    # Detect Cars
    for i in range(1, 5):
        scale = 1 + (0.1*i)
        cars = car_cascade.detectMultiScale(gray, scale, 3)
        car_array.append(cars)

    # Add to Heatmap
    for cars in car_array:
        for(x, y, w, h) in cars:
            heatmap[y:y+h, x:x+w] +=1

    '''-------------------
        Heatmap
    ----------------------'''
    heatmap_array.append(heatmap)
    smooth_heatmap = np.sum(heatmap_array[-10:], axis=0)

    if len(smooth_heatmap) <= min_threshold:
        threshold = len(smooth_heatmap) + 4
    else:
        threshold = min_threshold + 4
    smooth_heatmap = apply_threshold(smooth_heatmap, threshold=threshold)

    '''-------------------
        Labels
    ----------------------'''
    labels = label(smooth_heatmap)

    '''-------------------
        Draw Image
    ----------------------'''
    draw_img = draw_labeled_bboxes(draw_img, labels)

    return draw_img


'''============================
        Movie
==============================='''
from moviepy.editor import VideoFileClip

test_output = 'test_HAAR.mp4'
clip = VideoFileClip('project_video.mp4')
test_clip = clip.fl_image(process_frame)
test_clip.write_videofile(test_output, audio=False)

'''============================
        Images
==============================='''
# for img_src in test_images:
#     img = cv2.imread(img_src)
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     result = process_frame(img)
#     show(result)