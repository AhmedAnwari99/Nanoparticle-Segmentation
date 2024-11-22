import cv2 
import math

def sobel_filter(image):
    #Sobel Edge filter for enhancement of edges
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)  # X direction
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)  # Y direction

    # Compute the gradient magnitude (edge strength)
    sobel_magnitude = cv2.magnitude(sobel_x, sobel_y)

    # Convert the magnitude to 8-bit for adding to the original image
    sobel_magnitude_8bit = cv2.convertScaleAbs(sobel_magnitude)

    # Enhance the original image by adding the edge magnitude to it
    sobel_image = cv2.addWeighted((image), 1, sobel_magnitude_8bit, 1, 0)
    
    return sobel_image
        