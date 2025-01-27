import cv2
import yaml
import numpy as np
import matplotlib.pyplot as plt

from scipy.spatial import distance
from scipy.spatial.distance import directed_hausdorff

def get_config(config_filepath: str) -> dict:
    try:
        with open(config_filepath) as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        return {}

class NanoParticleSegmentation:
    def __init__(self):
        pass
        
    def get_gray_img(self, img_path):
        self.image = cv2.imread(img_path)
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        
        # Get the center of the image
        self.image_center = (self.image.shape[1] // 2, self.image.shape[0] // 2)

        return gray
        
    def contour_distance_to_center(self, contour):
        """
        Function to calculate the distance from the contour's centroid to the image center
        """
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            distance = np.sqrt((cX - self.image_center[0]) ** 2 + (cY - self.image_center[1]) ** 2)
            return distance
            
        return float('inf')

    def save_img(self, input_img, mask_img, seg_img, true_mask, img_save_path, metrics):
        # Display the results
        plt.figure(figsize=(10, 10))
        plt.suptitle(metrics, fontweight="bold", fontsize=16)

        plt.subplot(2, 2, 1)
        plt.imshow(cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB))
        plt.title("Input Image")
        plt.axis("off")

        plt.subplot(2, 2, 2)
        plt.title("True Mask")
        plt.imshow(true_mask, cmap="gray")
        plt.axis("off")

        plt.subplot(2, 2, 3)
        plt.title("Segmentation")
        plt.imshow(cv2.cvtColor(seg_img, cv2.COLOR_BGR2RGB))
        plt.axis("off")

        plt.subplot(2, 2, 4)
        plt.title("Pred Mask")
        plt.imshow(mask_img, cmap="gray")
        plt.axis("off")

        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to fit the suptitle


        plt.savefig(img_save_path, dpi=300, bbox_inches="tight")
        # plt.show()
        plt.close()

    def sort_contours(self, contours):
        # Sort contours first by distance to center, then by area
        contours_sorted = sorted(contours, key=lambda contour: (self.contour_distance_to_center(contour), -cv2.contourArea(contour)))
        
        # Select the largest contour closest to the center
        largest_cent_contour = contours_sorted[0] if contours_sorted else None

        return largest_cent_contour

    def segment(self, gray_img, strategy=1):
        # Silas' strategy
        if strategy == 1:       
            # Apply Gaussian Blur to reduce noise
            blurred = cv2.GaussianBlur(gray_img, (5, 5), 0)

            # Apply Median Blur to reduce noise
            blurred = cv2.medianBlur(gray_img, 5)

            # Thresholding using the mean value
            _, thresh = cv2.threshold(blurred, np.mean(blurred), 255, cv2.THRESH_BINARY_INV)
            
            # Apply Morphological Opening to remove small patches
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

        else:
            # Ahmed's strategy
            #Apply Median blur to remove salt and pepper noise
            blurred = cv2.medianBlur(gray_img, 9)

            #Thresholding to seperate particle from the background
            _, thresh_otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

            # Apply morphologicla operations
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            closed = cv2.morphologyEx(thresh_otsu, cv2.MORPH_CLOSE, kernel, iterations=3)

            kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel2, iterations=3)

        # Find contours in the processed image
        contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        largest_cent_contour = self.sort_contours(contours)
        
        # Create a blank mask and draw the largest contour
        mask = np.zeros_like(gray_img, dtype="uint8")
        if largest_cent_contour is not None:
            cv2.drawContours(mask, [largest_cent_contour], -1, (255), thickness=cv2.FILLED)
        
        # Fill small spaces in the mask
        filled_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        color = (0, 0, 255)
        thickness = 1
        
        # Draw contours of the refined segmentation mask on the original image in red
        seg_img = self.image.copy()
        refined_contours, _ = cv2.findContours(filled_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(seg_img, refined_contours, -1, color, thickness)
        
        # Draw red lines at the edges of seg_img
        height, width, _ = seg_img.shape
         
        # Top edge
        cv2.line(seg_img, (0, 0), (width, 0), color, thickness)
        # Bottom edge
        cv2.line(seg_img, (0, height - 1), (width, height - 1), color, thickness)
        # Left edge
        cv2.line(seg_img, (0, 0), (0, height), color, thickness)
        # Right edge
        cv2.line(seg_img, (width - 1, 0), (width - 1, height), color, thickness)
    
        return gray_img, filled_mask, seg_img


class Metrics:
    def __init__(self, mask, groundTruth):
        self.mask = mask
        self.groundTruth = groundTruth
        

    def calcDICE(self, smooth=1e-8):
        """
        Calculate the Dice similarity coefficient between two binary arrays.

        Parameters
        ----------
        mask : array_like
            Binary mask to compare against the ground truth.
        groundTruth : array_like
            Binary ground truth array.
        smooth : float, optional
            A very small value to add to the denominator to avoid division by zero.

        Returns
        -------
        dice : float
            The Dice similarity coefficient between the two arrays. It ranges from 0 (no similarity) to 1 (perfect similarity).

        Notes
        -----
        The Dice similarity coefficient is a measure of the overlap between two binary arrays. It is commonly used to evaluate the performance of image segmentation algorithms.

        References
        ----------
        .. [1] Dice, L. R. (1945). Measures of the Amount of Ecologic Association Between Species. Ecology, 26(3), 297-302.
        """
        mask = (self.mask > 0).astype(int)
        groundTruth = (self.groundTruth > 0).astype(int)
        dissimilarity = distance.dice(mask.flatten(), groundTruth.flatten())
        return 1 - dissimilarity

    def calcIOU(self):
        """
        Calculate the Intersection over Union (IoU) between two binary masks.

        Parameters
        ----------
        mask : array_like
            Binary mask to compare against the ground truth.
        groundTruth : array_like
            Binary ground truth array.

        Returns
        -------
        float
            The Intersection over Union (IoU) value, ranging from 0 (no overlap) to 1 (perfect overlap).

        Notes
        -----
        IoU is a common evaluation metric for image segmentation tasks, representing the area of overlap between the predicted mask and the ground truth divided by the area of their union.
        """
        intersection = np.logical_and(self.mask.flatten(), self.groundTruth.flatten())
        union = np.logical_or(self.mask.flatten(), self.groundTruth.flatten())
        return intersection.sum() / union.sum()

    def calcHusdorf(self):
        """
        Calculate the Hausdorff distance between two binary masks.

        Parameters
        ----------
        mask : array_like
            Binary mask to compare against the ground truth.
        groundTruth : array_like
            Binary ground truth array.

        Returns
        -------
        float
            The Hausdorff distance between the two masks.

        Notes
        -----
        The Hausdorff distance is a measure of the distance between two sets, in this case, the two binary masks. It is defined as the maximum distance between a point in one set and the closest point in the other set.
        """
        coords1 = np.column_stack(np.where(self.mask > 0))
        coords2 = np.column_stack(np.where(self.groundTruth > 0))
        return max(directed_hausdorff(coords1, coords2)[0], directed_hausdorff(coords2, coords1)[0])

    def get_metrics(self):
        return f"DICE: {self.calcDICE():.2f}, IoU: {self.calcIOU():.2f}, Hausdorf: {self.calcHusdorf():.2f}"