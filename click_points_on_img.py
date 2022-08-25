# importing the module
import os

import cv2


# function to display the coordinates of
# of the points clicked on the image
def click_event(event, x, y, flags, params):
    # checking for left mouse clicks
    if event == cv2.EVENT_LBUTTONDOWN:
        points_list.append([x,y])


    # checking for right mouse clicks
    if event == cv2.EVENT_RBUTTONDOWN:
        points_list.append([x, y])


if __name__ == "__main__":
    # reading the image
    img_dir = 'G:\Python projects\Horizon_Detection\data\img_data/'
    points_list = []
    for img_file in os.listdir(img_dir)[:3]:
        cv2.namedWindow('img')
        cv2.setMouseCallback('img', click_event)
        img = cv2.imread(img_dir + img_file, 1)
        while True:
            cv2.imshow('img', img)
            # Wait, and allow the user to quit with the 'esc' key
            k = cv2.waitKey(1)
            # If user presses 'esc' break
            if k == 27: break
        cv2.destroyAllWindows()


    print(points_list)
