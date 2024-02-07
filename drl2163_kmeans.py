import cv2
import os
import time
import numpy as np

#function to select which contour to use by counting number of white pixels
def sum_pixels(cont_lst, img):
    max_count = 0
    best_rect = None
    for contour in cont_lst:
        x, y, w, h = cv2.boundingRect(contour)
        roi = img[y:y + h, x:x + w]
        pix_count = np.count_nonzero(roi == 255)
        if pix_count > max_count:
            max_count = pix_count
            best_rect = [x, y, w, h]
    return max_count, best_rect


def get_box(img):
    #read in image to bgr and hsv spaces
    hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    #define lower and upper bounds for first red mask
    lower_red1 = np.array([0, 120, 70])
    upper_red1 = np.array([10, 255, 255])
    #define lower and upper bounds for second red mask
    lower_red2 = np.array([170, 120, 70])
    upper_red2 = np.array([180, 255, 255])
    #create both red masks and then combine them
    red_mask1 = cv2.inRange(hsv_image, lower_red1, upper_red1)
    red_mask2 = cv2.inRange(hsv_image, lower_red2, upper_red2)
    combined_red = cv2.bitwise_or(red_mask1, red_mask2)

    #pre process img and then set parameters and run k-means
    pixels = combined_red.reshape((-1, 1)).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    #using k = 2 because running k-means on masked image
    k = 2
    _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    #reshape labels back to original image
    labels = labels.reshape(img.shape[:2])

    #find contours
    stop_sign_region0 = (labels == 0).astype(np.uint8)
    stop_sign_region1 = (labels == 1).astype(np.uint8)
    contours0, _ = cv2.findContours(stop_sign_region0, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours1, _ = cv2.findContours(stop_sign_region1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    #get rectangle coordinates
    pix_count0, rect0 = sum_pixels(contours0, combined_red)
    pix_count1, rect1 = sum_pixels(contours1, combined_red)

    #choose correct coordinates and return them
    x0, y0, w0, h0 = rect0
    x1, y1, w1, h1 = rect1
    if (w0 * h0) > (w1 * h1):
        return x1, y1, (x1 + w1), (y1 + h1)
    else:
        return x0, y0, (x0 + w0), (y0 + h0)


if __name__ == "__main__":

    start_time = time.time()

    dir_path = './images/'
    for i in range(1, 25):
        img_name = f'stop{i}.png'
        img_path = os.path.join(dir_path, img_name)
        img = cv2.imread(img_path)
        # Get the coordinators of the box
        xmin, ymin, xmax, ymax = get_box(img)
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
        output_path = f'./results/{img_name}'
        cv2.imwrite(output_path, img)

    end_time = time.time()
    # Make it < 30s
    print(f"Running time: {end_time - start_time} seconds")

