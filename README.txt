SHORT TERM:
2. Replace 2 YOLOs in sequence with 1 trained for 3 classes (new weights required)
3. Cracks module
4. Dumpers module
5. Add new class - wooden poles (new weights required)
6. Put each block (1-2,3) including frame loader and results processor into separate threads connected with Queues in order to boost
system's performance
7. Subtract other object's boxes from the pillar BB (less noise for cracks detection)
8. Tune YOLOs parameters (NMS, threshold etc)

LONG TERM:
7. Batch processing to further speed up the system
8. Combine blocks 1-2 to decrease the number of expensive CPU-GPU data transfers (Apache Arrow to keep data on the local server)
9. New detections representation (single accomodating all detected objects alongside the frame). Now each subimage has the whole frame
matrix as an attribute, which takes a lot of space. We need one reference for all detected objects that then can crop it


KNOWN ISSUES:
- Server crashes after 1 request, doesn't exit to the state when its ready to process
another request

- BBs modification. Overlapping check.
- Class names and accuracy once written look bad
- 2 pillars can potentially get generated

There are requirements about how images need to be taken in order for the inclination detecting algorithm to process them.

------------------------------------------------
1. Test everything for images.
    1.1 Create N number of requests
    1.2 Nets initialize twice
    1.3 Server doesn't entirely exit after 1 request (some process's holding it)
    1.4 Test memory consumption

2. Fix threads
3. Add requirements