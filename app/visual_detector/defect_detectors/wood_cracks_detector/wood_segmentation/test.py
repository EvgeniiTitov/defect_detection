from app.visual_detector.defect_detectors.wood_cracks_detector.wood_segmentation import WoodCrackSegmenter
import os
import cv2



def main():
    classifier = WoodCrackSegmenter()

    path_to_images = r"D:\Desktop\system_output\Wood_Crack_Test"
    save_path = r"D:\Desktop\system_output\OUTPUT"
    images = list()

    for file in os.listdir(path_to_images):
        if not any(file.endswith(ext.lower()) for ext in ['.jpg', '.jpeg', '.png']):
            continue

        try:
            image = cv2.imread(os.path.join(path_to_images, file))
        except Exception as e:
            print(f"Failed while opening {file}. Skipped. Error: {e}")
            continue
        images.append(image)

    masked_images = classifier.process_batch(images)
    for i, masked_image in enumerate(masked_images):
        #masked_image.save(os.path.join(save_path, f"{i}_out.jpg"))
        cv2.imwrite(os.path.join(save_path, f"{i}_out.jpg"), masked_image)

if __name__ == "__main__":
    main()
