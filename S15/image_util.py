import cv2 
import numpy as np


def resize_image(image, dim):
    """
    Resizes images to the dimension given in dim
    """
    resizedImg = cv2.resize(image, dim)
    return resizedImg

def create_png_mask(image):
    """
    Creates mask based on alpha channel
    Sets transparency to black and non-transparent to white
    """
    B, G, R, A = cv2.split(image)
    
    h, w, c = image.shape
    maskedImg = np.zeros((h, w, 1), dtype=image.dtype)
    maskedImg[A != 0] = [255]

    return maskedImg


if __name__ == "__main__":
    for i in range(6):
        imgPath = "./fg/"+str(i)+".png"
        img = cv2.imread(imgPath, cv2.IMREAD_UNCHANGED)
        cv2.imshow("original", img)
        cv2.waitKey(0)
        height = 112
        width = 112
        resizedImage = resize_image(img, (width, height))
        cv2.imwrite("./fg_resized/"+str(i)+".png",resizedImage)
        
        maskedImg = resizedImage[:,:,3]
        cv2.imshow("masked", maskedImg)
        cv2.waitKey(0)
        cv2.imwrite("./fg_masked/"+str(i)+".jpg",maskedImg)
