
import numpy as np
import cv2 as cv
from tqdm import tqdm

def SSD(pixel_vals_1, pixel_vals_2):
                        #   Sum of squared distances for Correspondence
    if pixel_vals_1.shape != pixel_vals_2.shape:
        return -1

    return np.sum((pixel_vals_1 - pixel_vals_2)**2)

def window_comparison(y, x, block_left, right_array, window_size, x_search_window_size, y_search_window_size):
    # Block comparison function used for comparing windows on left and right images and find the minimum value ssd match the pixels
    
    # Get search range for the right image
    x_min = max(0, x - x_search_window_size)
    x_max = min(right_array.shape[1], x + x_search_window_size)
    y_min = max(0, y - y_search_window_size)
    y_max = min(right_array.shape[0], y + y_search_window_size)
    
    first = True
    min_ssd = None
    min_index = None

    for y in (range(y_min, y_max)):
        for x in range(x_min, x_max):
            block_right = right_array[y: y+window_size, x: x+window_size]
            ssd = SSD(block_left, block_right)
            if first:
                min_ssd = ssd
                min_index = (y, x)
                first = False
            else:
                if ssd < min_ssd:
                    min_ssd = ssd
                    min_index = (y, x)

    return min_index


def ssd_correspondence(img1, img2):
    """Correspondence applied on the whole image to compute the disparity map and finally disparity map is scaled"""
    # Don't search full line for the matching pixel
    # grayscale imges

    window_size = 15 # 15
    x_search_window_size = 50 # 50 
    y_search_window_size = 1
    h, w = img1.shape
    disparity_map = np.zeros((h, w))

    for y in tqdm(range(window_size, h-window_size)):
        for x in range(window_size, w-window_size):
            block_left = img1[y:y + window_size, x:x + window_size]
            index = window_comparison(y, x, block_left, img2, window_size, x_search_window_size, y_search_window_size)
            disparity_map[y, x] = abs(index[1] - x)
    
    disparity_map_unscaled = disparity_map.copy()

    # Scaling the disparity map
    max_pixel = np.max(disparity_map)
    min_pixel = np.min(disparity_map)

    for i in range(disparity_map.shape[0]):
        for j in range(disparity_map.shape[1]):
            disparity_map[i][j] = int((disparity_map[i][j]*255)/(max_pixel-min_pixel))
    
    disparity_map_scaled = disparity_map
    return disparity_map_unscaled, disparity_map_scaled


def disparitydepth(baseline, f, img):
    """This is used to compute the depth values from the disparity map"""

    # Assumption image intensities are disparity values (x-x') 
    depth_map = np.zeros((img.shape[0], img.shape[1]))
    depth_array = np.zeros((img.shape[0], img.shape[1]))

    for i in tqdm(range(depth_map.shape[0])):
        for j in range(depth_map.shape[1]):
            depth_map[i][j] = 1/img[i][j]
            depth_array[i][j] = baseline*f/img[i][j]
            # if math.isinf(depth_map[i][j]):
            #     depth_map[i][j] = 1

    return depth_map, depth_array