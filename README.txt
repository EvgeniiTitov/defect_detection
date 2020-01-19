To launch:
python defect_detection --image=..., --save_path=..., --crop_path=..., --pole_defects=1
python defect_detection --folder=..., --save_path=..., --crop_path=..., --pole_defects=1
python defect_detection --video=..., --save_path=..., --crop_path=..., --pole_defects=1

Now supports multiple input types --image=path_to_image --folder=path_to_folder

Known issues:
- BBs modification. Overlapping check.
- Components to the top can get cut by the pole's BBs. 0.9 * 2000 and 0.9 * 20 gives different effect!
Often the same happens with dumpers on both sides! For one pole detected it makes sense to widen
the box even more.


Main logic:

1. Parse user input, initialize things, create opencv window etc.
2. If images/folder open image and send it to object_detetion function
   along side the instance of the Neural Network class.
   If video it gets represented as VideoCapture class object. Send it to
   the obg_detection function along side video writer and neural net
3. Detections get stored in a dictionary where keys - section of the image
   in which detection takes place (whole image at first, then section containing
   a utility pole for components detection etc), values - objects detected
   on those image sections.
4. frame counter and while loop are required to process videos, keep looping till
   have frames while counting each frame. Required for saving frames (naming them)

   BLOCK 1
5. Create instance of the DetectionSection class with two arguments: the whole image
   (frame) and a name describing what we are looking for
6. Feed the frame/image to the block1 neural net detecting and classifying utility poles,
   save the poles if any have been detected.

   BLOCK 2
7. Check if any poles have been detected. In no poles found, send the whole image to the
   components detecting net in case there are close-up components on the image
   Otherwise, for each pole detected on the image form a new numpy matrix based on a pole's
   BBs representing a new section of the image on which detection of components will take place.
9. Create a new instance of the class DetectionSection with the newly created subimage.
   Remember coordinates of this subimage relatively to the original image to draw BBs
   at the end
10. Depending on the pole's class apply different components detecting neural nets (for
    concrete poles apart from insulator and dumper we also want to detect poles themselves)
11. If found any insulators perform insulator NORMALIZATION (TBD)
12. If any components found save them to the dictionary with the appropriate key (class
    instance representing image section on which those components were detected)

    BLOCK 3
    TBD

    DATABASE
    TBD

13. Crop out and save objects if any detected
14. Draw bounding boxes around the objects found
15. If working with video - save the processed frame
16. Show the frame/image
17. If working with images (len(image)>0) break out of the while loop. For videos repeat and
    check if there are frames left to process.