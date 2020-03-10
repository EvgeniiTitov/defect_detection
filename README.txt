SHORT TERM:
0. Line generating function needs to be timed out after N seconds
1. APIs
2. Combine 2 YOLOs into 1 (new weights required)
3. Cracks module
4. Dumpers module
5. Add new class - wooden poles (new weights required)
6. Put each block (1,2,3) including frame loader and results processor into separate threads connected with Queues in order to boost
system's performance

LONG TERM:
7. Batch processing to further speed up the system
8. Combine blocks 1-2 to decrease the number of expensive CPU-GPU data transfers (Apache Arrow to keep data on the local server)
9. New detections representation (single accomodating all detected objects alongside the frame)


Known issues:
- BBs modification. Overlapping check.
- Class names and accuracy once written look bad
- 2 pillars can potentially get generated

There are requirements about how images need to be taken in order for the inclination detecting algorithm to process them.