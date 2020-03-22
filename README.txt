SHORT TERM:
1. Retrain poles net to work with 3 class (add wood)
2. Replace 2 YOLOs in sequence with 1 trained for 3 components classes (new weights required)
3. Cracks module
4. Dumpers module
5. Subtract other object's boxes from the pillar BB (less noise for cracks detection)
6. Tune YOLOs parameters (NMS, threshold etc)

LONG TERM:
7. Batch processing to further speed up the system
8. Combine blocks 1-2 to decrease the number of expensive CPU-GPU data transfers (Apache Arrow to keep data on the local server)
9. New detections representation (single accomodating all detected objects alongside the frame). Now each subimage has the whole frame
matrix as an attribute, which takes a lot of space. We need one reference for all detected objects that then can crop it


KNOWN ISSUES:
- BBs modification. Overlapping check.
- Class names and accuracy once written look bad
- 2 pillars can potentially get generated
- Not well optimized line filtering algorithm


------------------------------------------------
1. Test everything for images.
    1.2 Nets initialize twice
    1.3 Sometimes server doesn't entirely exit after 1 request (some process's holding it)
    1.4 Test memory consumption
    1.5 How to kill first threads if threads after crash?

2. Fix threads
    2.1 Test video

3. Add requirements