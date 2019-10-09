class DetectedObject:
    """
    Represents an object detected. Keeps its coordinates, class name, confidence
    """
    def __init__(self, class_id, confidence, left, top, right, bottom):
        self.class_id = class_id
        self.confidence = confidence
        self.BB_top = top
        self.BB_left = left
        self.BB_right = right
        self.BB_bottom = bottom
        # Another set of coordinates to widen BB of metal pole's detected to address 
        # the insulators sticking out issue. By default they are the same
        self.top = top
        self.left = left
        self.right = right
        self.bottom = bottom

    def update_object_coordinates(self, left, top, right, bottom):
        """
        If a pole detected is a metal pole. Widen (probably even heighten) coordinates
        of this object to address the issue when insulators sticking out horizontally 
        do not get included in the object's bounding box, so they don't get detected by
        subsequent neural nets.
        """
        self.top = top
        self.left = left
        self.right = right
        self.bottom = bottom

    def __str__(self):
        return "Object detected: [{}, {}, {}, {}, {}, {}]".format(
                self.class_id, self.confidence, self.BB_left,
                self.BB_top, self.BB_right, self.BB_bottom)
        
        
class DetectionSection:
    """
    Represents part, section of the image in which detection takes place. Used to
    save objects detected by neural networks with reference to the image section in
    which the detection took place. For instance block 1 neural network uses the
    whole full size image. Block 2 neural nets do detection on the objects found
    by the block1 nets. We need this referencing for proper result handling at the end
    """
    def __init__(self, frame, name):
        self.frame = frame
        self.name = name
        self.top = 0
        self.left = 0
        self.right = 0
        self.bottom = 0
        
    def save_relative_coordinates(self, top, left, right, bottom):
        """
        When it comes to drawing BBs and saving objects detected in order to do so with
        the components, we need to know their relative positive relatively not to the
        original image, but rather to the image section in which they were detected - on poles 
        """
        self.top = top
        self.left = left
        self.right = right
        self.bottom = bottom
        
    def __str__(self):
        return f"Image section for searching: {self.name}, its size: {self.frame.shape}"
