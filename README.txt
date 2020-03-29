SHORT TERM TASKS:
1. Retrain poles net to work with 3 class (add wood)
2. Retrain components net to replace 2 YOLOs in sequence with 1 trained for 3 components classes
3. Cracks module
4. Dumpers module
5. Subtract other object's boxes from the pillar BB (less noise for cracks detection)
6. Tune YOLOs parameters (NMS, threshold etc)
7. Tune line filtering algorithm
8. Complete threads for video processing (
                                - actual code
                                - memory consumption to tune Qs,
                                - Method to shut down the server to .join() threads)

9. Separate API endpoint to process a video and return its angle



LONG TERM TASKS:
1. Batch processing to further speed up the system
2. Combine blocks 1-2 to decrease the number of expensive CPU-GPU data transfers (Apache Arrow to keep data on the local server)
3. New detections representation (single accomodating all detected objects alongside the frame). Now each subimage has the whole frame
matrix as an attribute, which takes a lot of space. We need one reference for all detected objects that then can crop it


KNOWN ISSUES:
- BBs modification. Overlapping check.
- Class names and accuracy once written look bad
- 2 pillars can potentially get generated
- Not well optimized line filtering algorithm


------------------------------------------------
1 Rewrite threads in accordance with Garlic's advices
2 Test memory consumption to tune Qs

3 Add method to join threads, shut down the whole server
4 How to get information about defects from the thread, without blocking it?


ASK OLEG:
1. Can we do polling to display progress?  or websockets
2.