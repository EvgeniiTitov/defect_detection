from app.visual_detector.defect_detectors.object_detector.detections_repr import DetectedObject
from typing import List, Tuple, Dict
from app.visual_detector.defect_detectors.object_detector.yolo.yolo import YOLOv3
from app.visual_detector.utils import TensorManager
import numpy as np
import os
import torch


class ComponentsDetector:
    """
    A wrapper around a neural network to do preprocessing / postprocessing
    Weights: Try 10 components
    """
    path_to_dependencies = r"C:\Users\Evgenii\Desktop\Python_Programming\Python_Projects\defect_detection\app\visual_detector\dependencies"
    dependencies_comp = "components"
    confidence = 0.10
    NMS_thresh = 0.25
    net_res = 608

    def __init__(self):
        # Initialize components predictor
        config_path_comp = os.path.join(self.path_to_dependencies, self.dependencies_comp + ".cfg")
        weights_path_comp = os.path.join(self.path_to_dependencies, self.dependencies_comp + ".weights")
        classes_path_comp = os.path.join(self.path_to_dependencies, self.dependencies_comp + ".txt")
        # Initialize detector - neural network
        try:
            self.components_net = YOLOv3(
                config=config_path_comp,
                weights=weights_path_comp,
                classes=classes_path_comp,
                confidence=ComponentsDetector.confidence,
                NMS_threshold=ComponentsDetector.NMS_thresh,
                network_resolution=ComponentsDetector.net_res
            )
            print("Components detector successfully initialized")
        except Exception as e:
            print(f"Failed during Component Detector initialization. Error: {e}")
            raise

    def process_batch(self, images_on_gpu: torch.Tensor, towers_predictions: Dict[int, List[DetectedObject]]) -> dict:
        """
        Receives a batch of images already loaded to GPU and poles detected on them
        Detects components within tower bounding boxes
        :param images_on_gpu:
        :param towers_predictions:
        :return:
        """
        '''
        Input format of detected towers: 
        {
            0: [DetectedObject1 (tower), DetectedObject2 (tower)...]}, 
            1: [DetectedObject3 (tower)...]}, 
            ...
        }
        For N images in the batch, we get M detected towers. On each tower up to Z components can be detected. 
        1. Slice out all detected towers, bring them all to one size and .cat() them. We need to
           remember how many towers were detected on each image in the batch.
        2. For M towers you get Z component detection. Each detection is to be represented as DetectedObject
        3. Perform matching. Distribute Z detected components among M towers belonging to N images.
        '''
        # Collect all images that will be used to search for components + how many towers found on each image
        imgs_to_search_components_on, distribute_info = self.collect_imgs(
            images_on_gpu=images_on_gpu,
            towers=towers_predictions
        )
        # Resize all collected images to the expected size of the net. Remember original sizes for bb rescaling
        resized_images, original_sizes = self.resize_imgs(imgs_to_search_components_on)

        # Concat all resized images in one tensor
        try:
            batch_search_components = torch.cat(resized_images)
        except Exception as e:
            print(f"Failed while .cat()ing images to search components on. Error: {e}")
            raise e
        # Normalize all images in the batch as per YOLO requirements
        batch_search_components.div_(255.0)

        # Run net, get component detections
        comp_detections = self.components_net.process_batch(images=batch_search_components)

        # Recalculate bbs relatively
        rescaled_detections = TensorManager.rescale_bounding_box(
            detections=comp_detections,
            current_dim=ComponentsDetector.net_res,
            equal_origin_shape=False,
            original_shapes=original_sizes
        )
        '''
        Postprocess results - represent all detections as class objects for convenience and add them to the same
        dictionary where detection results for towers have been stored.
        
        For each image N towers have been detected. This information is stored in distribute_info. 
        
        Let's say we have 5 frames. On 5 frames, 5 towers were detected on 4 images (1 had 2), there were no detections
        on one image. => Once combined, there're 5 + 1 images on which components will be detected because if no 
        towers have been found, we attempt to search component on the entire frame. 
        For each of the 6 images we will either get some or no component detections {1: [], 2: [], ...}
        
        We need to match (distrubute) C component detections among B tower detected on A images using the distribute
        information, which lists how many towers were found on each frame in the batch
        
        Match component detections with images on which the search happened (towers / entire frames). 
        '''
        detections_output = {i: list() for i in range(len(images_on_gpu))}
        matched_index = 0
        for i in range(len(images_on_gpu)):
            # Get number of towers that were detected on the i-th image in the batch
            index, nb_of_towers = distribute_info[i]
            # Assert image's index in the batch corresponds to the image index from distr. info
            assert i == index, "Indices do not match. Cannot perform detection matching"
            # If no towers were detected on i-th image in the batch, then we attempted to search for components on
            # the entire frame. Check if anything was detected
            if nb_of_towers == 0:
                # Loop over each detected component representing it as a class object for convenience
                components = rescaled_detections[matched_index]
                for component in components:
                    if component[-1] == 0:
                        class_name = "insulator"
                    elif component[-1] == 1:
                        class_name = "dumper"
                    elif component[-1] == 2:
                        class_name = "pillar"
                    else:
                        print(f"ERROR: Wrong class index got detected: {component[-1]}")
                        continue
                    try:
                        comp_obj = DetectedObject(
                            left=component[0],
                            top=component[1],
                            right=component[2],
                            bottom=component[3],
                            class_id=component[-1],
                            object_name=class_name,
                            confidence=component[5]
                        )
                    except Exception as e:
                        print(f"Failed during DetectedObject initialization. Error: {e}")
                        raise e
                    detections_output[i].append(comp_obj)

                del rescaled_detections[matched_index]
                matched_index += 1
                continue

            for j in range(nb_of_towers):
                components = rescaled_detections[matched_index]
                if not components:
                    del rescaled_detections[matched_index]
                    matched_index += 1
                    continue

                # Access each j-th tower object for i-th frame in order to save its relative coordinates
                tower_obj = towers_predictions[i][j]
                # Loop over each detected component representing it as a class object for convenience
                for component in components:
                    if component[-1] == 0:
                        class_name = "insulator"
                    elif component[-1] == 1:
                        class_name = "dumper"
                    elif component[-1] == 2:
                        class_name = "pillar"
                    else:
                        print(f"ERROR: Wrong class index got detected: {component[-1]}")
                        continue

                    #TODO: Pillar bb modification step to be implemented either here, or right before searching for
                    #      cracks/determining tilt angle.

                    try:
                        '''
                        While creating a DetectedObject instance for each component, make sure their bb coordinates
                        are relative to the original frame and not the object (tower) within which they were detected
                        '''
                        comp_obj = DetectedObject(
                            left=int(component[0] + tower_obj.left),
                            top=int(component[1] + tower_obj.top),
                            right=int(component[2] + tower_obj.left),
                            bottom=int(component[3] + tower_obj.top),
                            class_id=component[-1],
                            object_name=class_name,
                            confidence=component[5]
                        )
                    except Exception as e:
                        print(f"Failed during DetectedObject initialization. Error: {e}")
                        raise e

                    detections_output[i].append(comp_obj)
                del rescaled_detections[matched_index]
                matched_index += 1

        assert len(rescaled_detections) == 0, "Failed to match all component detections results."

        return detections_output

    def resize_imgs(self, images: List[torch.Tensor]) -> Tuple[list, list]:
        """
        Resizes provided images to the expected net size. Remembers information regarding original sizes
        of the images, which will be required to rescale bb
        :param images:
        :return:
        """
        resized_images, original_sizes = list(), list()

        for i in range(len(images)):
            image = images[i]
            try:
                resized_image, original_size = TensorManager.resize_tensor_keeping_aspratio(
                    batch_tensor=image.unsqueeze(0),
                    new_size=ComponentsDetector.net_res
                )
            except Exception as e:
                print(f"Failed during image resizing. Error: {e}")
                raise e

            resized_images.append(resized_image)
            original_sizes.append((i, original_size))

        return resized_images, original_sizes

    def collect_imgs(self, images_on_gpu: torch.Tensor, towers: dict) -> Tuple[list, list]:
        """
        Collects images on which components detection will take place.
        If any towers have been detected, modify their boxes (enlarge) to make sure all components end up
        within the tower's bounding box.
        :param images_on_gpu: batch of images on gpu
        :param towers: detected towers
        :return:
        """
        imgs_to_search_components_on = list()
        distrib_info = list()
        '''
        Check each frame in the batch and if any towers were found on it, slice out the tower's bb tensor
        If no towers found on the image, add the entire image to the list of images on which
        components will be detected 
        '''
        assert len(images_on_gpu) == len(towers), "ERROR: N of imgs in the batch != N of tower detection keys"
        for i in range(len(images_on_gpu)):
            # Get detections for an image in the batch and the image itself
            detections = towers[i]
            image = images_on_gpu[i]

            if not len(detections) > 0:
                # If no towers found on an image, take the entire image
                imgs_to_search_components_on.append(image)
                # 0 towers found for i-th image in the batch
                distrib_info.append((i, 0))
                continue

            nb_of_towers = len(detections)
            for detection in detections:
                # Modify object (tower)'s bounding boxes
                TensorManager.modify_bb_coord(tower=detection, image=image, nb_of_towers=nb_of_towers)
                # Get modified BB coordinates and slice out the tower
                left = detection.left
                top = detection.top
                right = detection.right
                bot = detection.bottom
                tower_bb_image = TensorManager.slice_out_tensor(image, [left, top, right, bot])
                imgs_to_search_components_on.append(tower_bb_image)

            # Keep track of how many towers detected on the i-th image in the batch
            distrib_info.append((i, len(detections)))

        return imgs_to_search_components_on, distrib_info

    def modify_pillars_bbs(self, image: np.ndarray, componenets_detected: dict) -> None:
        """
        Slightly widens pillar's bb in order to ensure both edges are within the box
        :param image:
        :param componenets_detected:
        :return:
        """
        for window, components in componenets_detected.items():
            for component in components:
                if component.class_id == 2:
                    new_left = component.BB_left * 0.96
                    new_right = component.BB_right * 1.04 if component.BB_right * 1.04 <\
                                                             image.shape[1] else image.shape[1] - 10
                    component.update_object_coordinates(left=int(new_left), right=int(new_right))

        return
