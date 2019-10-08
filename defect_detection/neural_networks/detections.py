class Detected_Object:
    '''
    Represents an object detected
    '''
    def __init__(self, class_id, confidence, left, top, right, bottom):
        self.class_id = class_id
        self.confidence = confidence
        self.BB_left = left
        self.BB_top = top
        self.BB_right = right
        self.BB_bottom = bottom
        # To widen BB of metal pole's detected to address insulators sticking out
        self.left = None
        self.top = None
        self.right = None
        self.bottom = None

    def __str__(self):
        return "Object detected: [{}, {}, {}, {}, {}, {}]".format(
                self.class_id, self.confidence, self.BB_left, 
                self.BB_top, self.BB_right, self.BB_bottom)

    def update_object_coordinates(self, left, top, right, bottom):
        '''
        If a pole detected is a metal pole. Widen (probably even heighten) coordinates
        of this object to address the issue when insulators sticking out horizontally 
        do not get included in the object's bounding box.
        '''
        self.left = left
        self.top = top
        self.right = right
        self.bottom = bottom
