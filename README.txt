SHORT TERM TASKS:
1. Retrain poles net to work with 3 class (Make sure classes are balanced)
2. Retrain components net to replace 2 YOLOs in sequence with 1 trained for 3 components classes (Balance dataset)
3. Dumpers module (Wait for Leonid to augment data, try teaching)
4. Wooden pole defects (Dasha)
5. Tuning:
    - Tune YOLOs parameters (NMS, threshold etc)
    - Tune line filtering algorithm
    - Tune Q sizes (measure memory consumptions etc)


6. Add and connect SQL to save defects properly
! 7. Fix the issue (remove frame from all representation objects, you have access to it anyway in the workers)
! 8. Inference 1 in N frames - fix your issue when for videos only 1 frame results are saved


LONG TERM TASKS:
1. Batch processing to further speed up the system
2. Combine blocks 1-2 to decrease the number of expensive CPU-GPU data transfers (Apache Arrow to keep data on the local server)
3. New detections representation (single accomodating all detected objects alongside the frame). Now each subimage has the whole frame
matrix as an attribute, which takes a lot of space. We need one reference for all detected objects that then can crop it
- FROZEN. Cracks module
- FROZEN. Subtract other object's boxes from the pillar BB (less noise for cracks detection)


KNOWN ISSUES:
- You pass frame between workers whereas its already stored inside each detected pole object
- BBs modification. Overlapping check.
- Class names and accuracy once written look bad
- 2 pillars can potentially get generated
- Not well optimized line filtering algorithm


14.04
! 7
! 8
Relisten talk to Igor
Review and complete your webdev course