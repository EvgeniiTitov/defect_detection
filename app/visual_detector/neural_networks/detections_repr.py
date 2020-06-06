class DetectedObject:
    """
    Represents an object detected. Keeps its coordinates, class ID, confidence score
    """
    def __init__(
            self,
            class_id,
            confidence,
            left,
            top,
            right,
            bottom,
            object_name=None
    ):
        self.class_id = class_id
        self.object_name = object_name
        self.confidence = confidence

        # Original bb coordinates to draw bounding boxes
        assert all((left >= 0, top >= 0, right >= 0, bottom >= 0)), "ERROR: Negative BB coordinate(s) provided"
        assert all((left < right, top < bottom)), "ERROR: Incorrect BB coordinates provided"
        self.BB_top = top
        self.BB_left = left
        self.BB_right = right
        self.BB_bottom = bottom

        # Second set of coordinates to be able to modify them
        self._top = top
        self._left = left
        self._right = right
        self._bottom = bottom

        # Deficiency information
        self.deficiency_status = False
        self.inclination = None
        self.cracked = None

    @property
    def top(self):
        return self._top

    @property
    def left(self):
        return self._left

    @property
    def right(self):
        return self._right

    @property
    def bottom(self):
        return self._bottom

    def update_object_coordinates(
            self,
            left=None,
            top=None,
            right=None,
            bottom=None
    ):
        """
        Modifies object's BBs (mainly applies to utility poles detected) to make sure
        when searching for this object's components nothing gets missed because it didn't
        end up in the object's BB.
        """
        # Update value only of the coordinates provided. Doesn't change default box values
        # which are equal to the values of the BBs predicted.
        if left and right:
            assert left >= 0 and right >= 0 and left < right, "Incorrect coordinates provided"
            self._left = left
            self._right = right
        elif left:
            assert left >= 0, "Incorrect coordinate provided"
            self._left = left
        elif right:
            assert right >= 0, "Incorrect coordinate provided"
            self._right = right

        if top and bottom:
            assert top >= 0 and bottom >= 0 and top < bottom, "Incorrect coordinates provided"
            self._top = top
            self._bottom = bottom
        elif top:
            assert top >= 0, "Incorrect coordinates provided"
            self._top = top
        elif bottom:
            assert bottom >= 0, "Incorrect coordinates provided"
            self._bottom = bottom

    def __str__(self):
        return "Detected object. Class: {}, Conf: {}, Coord: {}, {}, {}, {}".format(
            self.class_id, self.confidence, self.BB_left, self.BB_top, self.BB_right, self.BB_bottom
        )
        
        
class SubImage:
    """
    Represents part, section of the image on which detection takes place. Is used to
    save objects detected by neural networks with reference to the image section on
    which the detection took place - Relative Coordinates.
    """
    def __init__(self, name: str):
        self.name = name
        self._top = 0
        self._left = 0
        self._right = 0
        self._bottom = 0

    @property
    def top(self):
        return self._top

    @property
    def left(self):
        return self._left

    @property
    def right(self):
        return self._right

    @property
    def bottom(self):
        return self._bottom

    def save_relative_coordinates(
            self,
            top: int,
            left: int,
            right: int,
            bottom: int
    ):
        """
        When it comes to drawing BBs and saving objects detected in order to do so with
        the components, we need to know their relative positive relatively not to the
        original image, but rather to the image section on which they were detected - on poles
        """
        if not all((top >= 0, left >= 0, right >= 0, bottom >= 0)):
            raise ValueError("Negative relative coordinate provided")
        self._top = top
        self._left = left
        self._right = right
        self._bottom = bottom

    def __str__(self):
        return "Subimage. Name: {}, Coordinates: {}, {}, {}, {}".format(
            self.name, self._top, self._left, self._right, self._bottom
        )
