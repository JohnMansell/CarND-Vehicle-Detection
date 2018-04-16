import os
import glob
import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import time
from skimage.feature import hog
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from scipy.ndimage.measurements import label
import pickle

'''-------------------------
        Feature Parameters
----------------------------'''
color_space = 'YUV'
orient = 9
pix_per_cell = 8
cell_per_block = 2
hog_channel = 'ALL'
spatial_size = (32, 32)
hist_bins = 200
spatial_feat = True
hist_feat = True
hog_feat = True

# Image Parameters
n_samples = 100
y_start_stop = [380, 620]
x_start_stop = [None, None]
overlap = 0.5
xy_window = (128, 128)

ystart = 400
ystop = 656
scale = 1.5

usePickle = True
make_video = False
offset = 12

count = 1
'''====================================
        Function Definitions
======================================='''
'''-------------------------
        HOG features
----------------------------'''
def get_hog_features(img, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True):

    # Call with two outputs if vis==True
    if vis == True:
        features, hog_image = hog(img,
                                  orientations=orient,
                                  pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block),
                                  transform_sqrt=True,
                                  visualise=vis,
                                  feature_vector=feature_vec)
        return features, hog_image

    # Otherwise call with one output
    else:
        features = hog(img,
                       orientations=orient,
                       pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block),
                       transform_sqrt=True,
                       visualise=vis,
                       feature_vector=feature_vec)
        return features

'''-------------------------
        Bin Spatial
----------------------------'''
def bin_spatial(img, size=spatial_size):
    color1 = cv2.resize(img[:, :, 0], size).ravel()
    color2 = cv2.resize(img[:, :, 1], size).ravel()
    color3 = cv2.resize(img[:, :, 2], size).ravel()

    return np.hstack((color1, color2, color3))

'''-------------------------
        Color Histogram
----------------------------'''
def color_hist(img, nbins=hist_bins, bins_range = (0,256)):

    bmin = np.min(img)
    bmax = np.max(img)

    # Compute the color histogram of the color channels separately
    channel1_hist = np.histogram(img[:, :, 0], bins=nbins, range=(bmin, bmax))
    channel2_hist = np.histogram(img[:, :, 1], bins=nbins, range=(bmin, bmax))
    channel3_hist = np.histogram(img[:, :, 2], bins=nbins, range=(bmin, bmax))

    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))

    return hist_features

'''-------------------------
        Extract Features
----------------------------'''
def extract_features(imgs,
                     color_space=color_space,
                     spatial_size=spatial_size,
                     hist_bins=hist_bins,
                     orient=orient,
                     pix_per_cell=pix_per_cell,
                     cell_per_block=cell_per_block,
                     hog_channel=hog_channel,
                     spatial_feat=spatial_feat,
                     hist_feat=hist_feat,
                     hog_feat=hog_feat):

    #Feature Vectors List
    features = []

    for file in imgs:
        file_features = []
        image = mpimg.imread(file)

        # Convert Color Space
        if color_space != 'RGB':
            if color_space == 'HSV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            elif color_space == 'LUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
            elif color_space == 'HLS':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
            elif color_space == 'YUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
            elif color_space == 'YCrCb':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)

        else:
            feature_image = np.copy(image)

        if spatial_feat is True:
            spatial_features = bin_spatial(feature_image, size=spatial_size)
            file_features.append(spatial_features)

        if hist_feat is True:
            hist_features = color_hist(feature_image, nbins=hist_bins)
            file_features.append(hist_features)

        if hog_feat is True:
            if hog_channel is 'ALL':
                hog_features = []
                for channel in range(feature_image.shape[2]):
                    hog_features.append(get_hog_features(feature_image[:, :, channel],
                                                         orient = orient,
                                                         pix_per_cell = pix_per_cell,
                                                         cell_per_block = cell_per_block,
                                                         vis=False,
                                                         feature_vec=True))
                hog_features = np.ravel(hog_features)
            else:
                hog_features = get_hog_features(feature_image[:, :, hog_channel],
                                                orient=orient,
                                                pix_per_cell=pix_per_cell,
                                                cell_per_block=cell_per_block,
                                                vis=False,
                                                feature_vec=True)
            file_features.append(hog_features)

        # Append all File Features
        features.append(np.concatenate(file_features))

    return features

'''-------------------------
        Slide Windows ##
----------------------------'''
def slide_window(img,
                 x_start_stop=[None, None],
                 y_start_stop=[None, None],
                 xy_window=xy_window,
                 xy_overlap=(overlap, overlap)):

    # If (x,y) start / stop not defined -- set to image size
    if x_start_stop[0] == None:
        x_start_stop[0] = 0

    if x_start_stop[1] == None:
        x_start_stop[1] = img.shape[1]

    if y_start_stop[0] == None:
        y_start_stop[0] = 0

    if y_start_stop[1] == None:
        y_start_stop[1] = img.shape[0]

    # Span of Region to be searched
    xspan = x_start_stop[1] - x_start_stop[0]
    yspan = y_start_stop[1] - y_start_stop[0]

    # Number of Pixels per step in x,y
    nx_pix_per_step = np.int(xy_window[0]*(1 - xy_overlap[0]))
    ny_pix_per_step = np.int(xy_window[1]*(1 - xy_overlap[1]))

    # Number of Windows in x,y
    nx_windows = np.int(xspan / nx_pix_per_step) - 1
    ny_windows = np.int(yspan / ny_pix_per_step) - 1

    # List of window positions
    window_list = []

    for ys in range(ny_windows):
        for xs in range(nx_windows):
            # Calculate Window Position
            startx = xs*nx_pix_per_step + x_start_stop[0]
            endx = startx + xy_window[0]

            starty = ys*ny_pix_per_step + y_start_stop[0]
            endy = starty + xy_window[1]

            window_list.append(((startx, starty), (endx, endy)))

    return window_list


'''-------------------------
        Draw Boxes
----------------------------'''
def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    imcopy = np.copy(img)

    for bbox in bboxes:
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)

    return imcopy

'''-------------------------
        Single Image Features
----------------------------'''
def single_img_features(img,
                        color_space=color_space,
                        spatial_size=spatial_size,
                        hist_bins=hist_bins,
                        orient=orient,
                        pix_per_cell=pix_per_cell,
                        cell_per_block=cell_per_block,
                        hog_channel=hog_channel,
                        spatial_feat=spatial_feat,
                        hist_feat=hist_feat,
                        hog_feat=hog_feat,
                        vis=False):
    img_features = []

    # Convert Color Space
    if color_space != 'RGB':
        if color_space == 'HSV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        elif color_space == 'LUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
        elif color_space == 'HLS':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        elif color_space == 'YUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        elif color_space == 'YCrCb':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)

    else: feature_image = np.copy(img)

    if spatial_feat is True:
        spatial_features = bin_spatial(feature_image, size=spatial_size)
        img_features.append(spatial_features)

    if hist_feat is True:
        hist_features = color_hist(feature_image, nbins=hist_bins)
        img_features.append(hist_features)

    if hog_feat is True:
        if hog_channel is 'ALL':
            hog_features = []
            for channel in range(feature_image.shape[2]):
                hog_features.append(get_hog_features(feature_image[:, :, channel],
                                                     orient = orient,
                                                     pix_per_cell = pix_per_cell,
                                                     cell_per_block = cell_per_block,
                                                     vis=False,
                                                     feature_vec=True))
            hog_features = np.concatenate(hog_features)
        else:
            if vis == True:
                hog_features, hog_image = get_hog_features(feature_image[:, :, hog_channel],
                                                           orient = orient,
                                                           pix_per_cell = pix_per_cell,
                                                           cell_per_block = cell_per_block,
                                                           vis=True,
                                                           feature_vec=True)
            else:
                hog_features = get_hog_features(feature_image[:, :, hog_channel],
                                                orient,
                                                pix_per_cell,
                                                cell_per_block,
                                                vis=False,
                                                feature_vec=True)

        img_features.append(hog_features)

    if vis == True:
        return np.concatenate(img_features), hog_image

    else:
        return np.concatenate(img_features)

'''-------------------------
        Search Windows
----------------------------'''
def search_windows(img, windows, clf, scaler, color_space='RGB',
                   spatial_size=spatial_size,
                   hist_bins=hist_bins,
                   hist_range=(0, 256),
                   orient=orient,
                   pix_per_cell=pix_per_cell,
                   cell_per_block=cell_per_block,
                   hog_channel=hog_channel,
                   spatial_feat=True,
                   hist_feat=True,
                   hog_feat=True):

    # 1) Create an empty list to receive positive detection windows
    on_windows = []
    heatmap = np.zeros_like(img)

    # 2) Iterate over all windows in the list
    for window in windows:
        test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))
        features = single_img_features(test_img,
                                       color_space=color_space,
                                       spatial_size=spatial_size,
                                       hist_bins=hist_bins,
                                       orient=orient,
                                       pix_per_cell=pix_per_cell,
                                       cell_per_block=cell_per_block,
                                       hog_channel=hog_channel,
                                       spatial_feat= spatial_feat,
                                       hist_feat = hist_feat,
                                       hog_feat=hog_feat)

        test_features = scaler.transform(np.array(features).reshape(1, -1))
        prediction = clf.predict(test_features)

        if prediction == 1:
            on_windows.append(window)
            # w1 = np.array(window[0])
            # w2 = np.array(window[1])
            #
            # x1 = w1[0]
            # y1 = w1[1]
            #
            # x2 = w2[0]
            # y2 = w2[1]
            #
            # heatmap[y1:y2, x1:x2] +=1

    return heatmap, on_windows

'''-------------------------
        Visualize
----------------------------'''
# def visualize(fig, rows, cols, imgs, titles):
#     for i, img in enumerate(imgs):
#         plt.subplot(rows, cols, i+1)
#         plt.title(i+1)
#         img_dims = len(img.shape)
#
#         if img_dims < 3:
#             plt.imshow(img, cmap='hot')
#             plt.title(titles[i])
#
#         else:
#             plt.imshow(img)
#             plt.title(titles[i])
#
#     plt.show()

'''-------------------------
        Convert Color
----------------------------'''
def convert_color(img, color_space=color_space, BGR = False):

    if BGR == False:
        if color_space == 'HSV':
            return cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        if color_space == 'LUV':
            return cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
        if color_space == 'YCrCb':
            return cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
        if color_space == 'RGB':
            return img
        if color_space == 'YUV':
            return cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        else:
            print("\nConvert Color -- Invalid Color Space", color_space)
            exit()

    else:
        if BGR == True:
            if color_space == 'HSV':
                return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            if color_space == 'LUV':
                return cv2.cvtColor(img, cv2.COLOR_BGR2LUV)
            if color_space == 'YCrCb':
                return cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
            if color_space == 'RGB':
                return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if color_space == 'YUV':
                return cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
            else:
                print("\nConvert Color -- Invalid Color Space", color_space)
                exit()


'''-------------------------
        Find Cars
----------------------------'''

# Define a single function that can extract features using hog sub-sampling and make predictions
def find_cars(img,
              svc,
              X_scaler,
              ystart = ystart,
              ystop = ystop,
              scale = scale,
              color_space = color_space,
              orient = orient,
              pix_per_cell = pix_per_cell,
              cell_per_block = cell_per_block,
              hog_channel = hog_channel,
              spatial_size = spatial_size,
              hist_bins = hist_bins,
              i = 1):

    if x_start_stop[0] == None:
        x_start_stop[0] = 0
    if x_start_stop[1] == None:
        x_start_stop[1] == img.shape[1]

    y1 = y_start_stop[0]
    y2 = y_start_stop[1]
    x1 = x_start_stop[0]
    x2 = x_start_stop[1]

    img_tosearch = img[y1:y2, x1:x2]
    heatmap = np.zeros_like(img[:, :, 0])
    img_boxes = []

    # Prepare Image -- mask and color conversion

    ctrans_tosearch = convert_color(img_tosearch, color_space = color_space)

    # Resize Image
    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1] / scale), np.int(imshape[0] / scale)))

    ch1 = ctrans_tosearch[:, :, 0]
    ch2 = ctrans_tosearch[:, :, 1]
    ch3 = ctrans_tosearch[:, :, 2]

    # Define blocks and steps as above
    nxblocks = (ch1.shape[1] // pix_per_cell) - 1
    nyblocks = (ch1.shape[0] // pix_per_cell) - 1
    nfeat_per_block = orient * cell_per_block ** 2
    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // pix_per_cell) - 1
    cells_per_step = 2 # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step

    # Compute individual channel HOG features for the entire image
    if hog_channel == 'ALL':
        hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
        hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
        hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)

    else:
        hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)

    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb * cells_per_step
            xpos = xb * cells_per_step
            # Extract HOG for this patch

            if hog_channel == 'ALL':
                hog_feat1 = hog1[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
                hog_feat2 = hog2[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
                hog_feat3 = hog3[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
                hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))
            else:
                hog_features = hog1[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()

            xleft = xpos * pix_per_cell
            ytop = ypos * pix_per_cell

            # Extract the image patch
            subimg = cv2.resize(ctrans_tosearch[ytop:ytop + window, xleft:xleft + window], (64, 64))

            if spatial_feat == True:
                spatial_features = bin_spatial(subimg, size=spatial_size)

            if hist_feat == True:
                hist_features = color_hist(subimg, nbins=hist_bins)

            # Scale features and make a prediction
            if ((spatial_feat == True) and (hist_feat == True)):
                stacked = np.hstack((spatial_features, hist_features, hog_features))

            elif((spatial_feat == True) and (hist_feat == False)):
                stacked = np.hstack((spatial_features, hog_features))

            elif((spatial_feat == False) and (hist_feat == True)):
                stacked = np.hstack((hist_features, hog_features))
            elif((spatial_feat == False) and (hist_feat == False)):
                stacked = hog_features

            reshaped = stacked.reshape(1, -1)
            test_features = X_scaler.transform(reshaped)

            # Predict car or not-car
            test_prediction = svc.predict(test_features)

            if test_prediction == 1:

                win_draw = np.int(window * scale)

                x1 = np.int(xleft * scale) + x_start_stop[0] - offset
                x2 = np.int(xleft * scale) + x_start_stop[0] + win_draw + offset

                y1 = np.int(ytop * scale) + y_start_stop[0] - offset
                y2 = np.int(ytop * scale) + y_start_stop[0] + win_draw + offset

                img_boxes.append(((x1, y1), (x2, y2)))
                heatmap[y1:y2, x1:x2] +=1
    return heatmap

'''-------------------------
        Apply Threshold
----------------------------'''
def apply_threshold(heatmap, threshold):
    heatmap[heatmap <= threshold] = 0
    return heatmap


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

'''-------------------------
        Mag Threshold
----------------------------'''
def mag_threshold(img, sobel_kernel=5, mag_thresh=(-1, 237)):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    gradmag = np.sqrt(sobelx ** 2 + sobely ** 2)
    scale_factor = np.max(gradmag) / 255
    binary_output = np.zeros_like(gradmag)

    binary_output[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1
    return binary_output


'''==========================
        Multiple Images
============================='''

'''-------------------------
    Read In Images
----------------------------'''

# Read in cars and notcars
cars = glob.glob('vehicles/*/*.png')
notcars = glob.glob('non-vehicles/*/*.png')
searchpath = 'test_images/*'
example_images = glob.glob(searchpath)

'''-------------------------
        Train Classifier
----------------------------'''
if usePickle == False:
    car_ind = np.random.randint(0, len(cars))
    notcar_ind = np.random.randint(0, len(notcars))

    # Read in car / notcar images
    car_image = mpimg.imread(cars[car_ind])
    notcar_image = mpimg.imread(notcars[notcar_ind])

    # Create Random Car indx's
    random_idxs = np.random.randint(0, len(cars), n_samples)
    test_cars = cars
    test_notcars = notcars

    '''-------------------------
            Train Classifier
    ----------------------------'''
    # Extract Features
    car_features = extract_features(test_cars)
    notcar_features = extract_features(test_notcars)

    # Stack features
    X = np.vstack((car_features, notcar_features)).astype(np.float64)

    # Fit a per-column scaler
    X_scaler = StandardScaler().fit(X)

    # Apply the scaler to X
    scaled_X = X_scaler.transform(X)

    # Define the labels vector
    y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

    # Split up data into randomized training and test sets
    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size=0.1, random_state=rand_state)

    print('Feature Vector Length :: ', len(X_train[0]))
    print('X - Train = ', X_train.shape)

    # Use a Linear SVC
    svc = LinearSVC()

    # Check training time for SVC
    svc.fit(X_train, y_train)

    # Check the score of the SVC
    print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))

'''-------------------------
        Save Pickle
----------------------------'''
if usePickle == False:
    with open('svc_pickle.pkl', 'wb') as fid:
        pickle.dump(svc, fid)
        pickle.dump(X_scaler, fid)

else:
    with open('svc_pickle.pkl', 'rb') as fid:
        svc = pickle.load(fid)
        X_scaler = pickle.load(fid)
        print("Pickle Loaded")


'''---------------------------------
        Process Image
------------------------------------'''
heatmaps = []

def process_img(img):

    draw_img = np.copy(img)
    img = img.astype(np.float32) / 255

    # Find Cars
    for i in range(4):
        scale = 1.0 + (0.2 * i)
        heatmap = find_cars(img, svc=svc, X_scaler=X_scaler, scale=scale, i=i)
        heatmaps.append(heatmap)

    '''-------------------
        Heatmap
    ----------------------'''
    if make_video == True:
        total_heatmap = np.sum(heatmaps[-21:], axis=0)
        if len(heatmaps) < 21:
            threshold = len(heatmaps) //3
        else:
            threshold = 7

        # Apply Threshold
        cooled_heatmap = apply_threshold(total_heatmap, threshold=threshold)

    if make_video == False:
        threshold = 3
        total_heatmap = np.sum(heatmaps[-3:], axis=0)
        cooled_heatmap = apply_threshold(total_heatmap, threshold=threshold)

    '''-------------------
        Labels
    ----------------------'''
    labels = label(cooled_heatmap)

    '''-------------------
        Draw Image
    ----------------------'''
    draw_img = draw_labeled_bboxes(draw_img, labels)
    return draw_img


'''=======================
    Test Images
=========================='''
count = 0
if make_video == False:
    for img_src in example_images:
        img = cv2.imread(img_src)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = process_img(img)
        count += 1
        title = "result" + str(count)
        show(result, title=title)

'''=======================
    Video
=========================='''
if make_video == True:
    from moviepy.editor import VideoFileClip

    test_output = 'project_output_3.mp4'
    clip = VideoFileClip('project_video.mp4')
    test_clip = clip.fl_image(process_img)
    test_clip.write_videofile(test_output, audio=False)




































































