To launch:
python defect_detection --image=..., --save_path=..., --crop_path=..., --pole_defects=1
python defect_detection --folder=..., --save_path=..., --crop_path=..., --pole_defects=1
python defect_detection --video=..., --save_path=..., --crop_path=..., --pole_defects=1

Now supports multiple input types --image=path_to_image --folder=path_to_folder

TO DO:
    0. Ensure system scalability - more classes, more defects to search for
    1. Defect detecting hub
        1.1 Bring all objects detected there and list of defects to look for (all
            by default)
        1.2 Depending on object's class select the appropriate pipeline, some may share
            first steps like inclination detection and cracks.
        1.3 Multiprocessing VS multiple GPUs VS typical order (very slow)
        1.4 How to store defects? XML vs JSON. Label BBs of defected objects in red.
    2. Bring over and integrate the inclination detection module with
       cracks detection in view
    3. Think how and where you will be saving files and defects (JSON vs XML)
    4. APIs
    5. Proper dependencies management (weights and cfg files)
    6. Combine 2 YOLOs into 1
    7. YOLO --> Torch
    8. Cracks module
    9. Dumpers module
    10. Rigorous tests
    11. Basic GUI

Known issues:
- BBs modification. Overlapping check.
- Class names and accuracy looks very bad
- Performance. Slow.