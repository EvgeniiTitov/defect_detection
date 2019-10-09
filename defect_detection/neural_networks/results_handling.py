import os

def save_objects_detected(objects_detected, save_path, path_to_image):
    """
    Function that saves objects detected by the neural networks on the disk
    :param objects: Dictionary with objects found
    :return: None
    """
    for image_section, objects in objects_detected.items():
        for i, object in enumerate(objects):
            cropped_frame = image_section.frame[object.top+image_section.top:object.bottom+image_section.top,
                                                object.left+image_section.left:object.right+image_section.left]
            object_name = 
    
    
    pass

def draw_bounding_boxes(objects):
    
    
    pass