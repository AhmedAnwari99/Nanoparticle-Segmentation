import os
import cv2
from natsort import natsorted
from utils import get_config, NanoParticleSegmentation, Metrics

def process(config_path="./config.yaml"):
    config = get_config(config_filepath=config_path)
    images_dir = config.get("images_dir", None)
    masks_dir = config.get("masks_dir", None)
    save_dir = config.get("save_dir", None)
    os.makedirs(save_dir, exist_ok=True)

    nano_particle_segmentor = NanoParticleSegmentation()

    image_names = [
        os.path.join(folder_name, name)
        for folder_name in os.listdir(images_dir)
        for name in natsorted(os.listdir(os.path.join(images_dir, folder_name)))
        if name.endswith(('.png', '.jpg', '.jpeg'))
    ]
    
    mask_names = [
        os.path.join(folder_name, name)
        for folder_name in os.listdir(masks_dir)
        for name in natsorted(os.listdir(os.path.join(masks_dir, folder_name)))
        if name.endswith(('.png', '.jpg', '.jpeg'))
    ]
    
    # Process images and masks
    for image_name, mask_name in zip(image_names, mask_names):
        img_path = os.path.join(images_dir, image_name)
        mask_path = os.path.join(masks_dir, mask_name)

        gray_img = nano_particle_segmentor.get_gray_img(img_path)
        true_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # gray_img, filled_mask, seg_img = nano_particle_segmentor.segment(gray_img)
        gray_img, filled_mask, seg_img = nano_particle_segmentor.segment(gray_img, strategy=2)

        metrics_generator = Metrics(filled_mask, true_mask)
        metrics = metrics_generator.get_metrics()

        img_save_path = os.path.join(save_dir, os.path.basename(image_name))
        nano_particle_segmentor.save_img(gray_img, filled_mask, seg_img, true_mask, img_save_path, metrics)

        print(f"File: {os.path.basename(image_name)} - Metrics: {metrics}")

if __name__ == "__main__":
    process()
