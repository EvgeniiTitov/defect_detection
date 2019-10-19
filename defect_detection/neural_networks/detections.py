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
        # Another set of coordinates to modify BBs of objects detected to achieve certain
        # things. For instance, to address an issue of insulators sticking out and not getting
        # detected because they do not get included in the pole's BB. Widen the BB in this case
        self.top = top
        self.left = left
        self.right = right
        self.bottom = bottom

    def update_object_coordinates(self, left=None, top=None, right=None, bottom=None):
        """
        Modifies object's BBs (mainly applies to utility poles detected) to make sure
        when searching for this object's components nothing gets missed because it didn't
        end up in the object's BB. So, we can either wider/heighten the BB or even tighten
        the box when it comes to detecting concrete pole for defect detection.
        Does not modifies BBs values that will be used for BBs drawing and cropping out objects
        """
        # Update value only of the coordinates provided. Doesn't change default box values
        # which are equal to the values of the BBs predicted.
        if top:
            self.top = top
        if left:
            self.left = left
        if right:
            self.right = right
        if bottom:
            self.bottom = bottom

    def __str__(self):
        return "Object: {}, {}, {}, {}, {}, {}".format(
                self.class_id, self.confidence, self.BB_left,
                self.BB_top, self.BB_right, self.BB_bottom)
        
        
class DetectionImageSection:
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
        original image, but rather to the image section on which they were detected - on poles
        """
        self.top = top
        self.left = left
        self.right = right
        self.bottom = bottom
        
    def __str__(self):
        return f"Image section for searching: {self.name}, its size: {self.frame.shape}"
