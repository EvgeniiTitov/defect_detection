SHORT TERM TASKS:
1. Retrain poles net to work with 3 class (Make sure classes are balanced)
2. Retrain components net to replace 2 YOLOs in sequence with 1 trained for 3 components classes (Balance dataset)
3. Dumpers module (Wait for Leonid to augment data, try teaching)
4. Wooden pole defects (Dasha)
5. Tuning:
    - Tune YOLOs parameters (NMS, threshold etc)
    - Tune line filtering algorithm
    - Tune Q sizes (measure memory consumptions etc) - nvidia-smi in console, look it up
    Если память не забивается достаточно, надо попробовать забить посильнее чем вечно с ХОСТА на
    ДЕВАЙС (cpu-gpu?) двигать.

6. BBs interpolation? (run actual nets once in N frames as well, just remmember coordinates and then
update them)
7. Object tracking (bb overlapping check suggested by Igor) - could be done right in the writer or another
worker before it.

9. Before serving the first request, we need to make sure NNs, threads are up and running. + DB connection
10. Ability to restart NNs and threads. Kill them, not actual server and then restart before first request


LONG TERM TASKS:
1. Batch processing to further speed up the system (currently only 40% of GPU is used)
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


SERVER:
One endpoint - sorter. N number of parallel pipelines that all have access to the database.
Neural networks folder should become subfolder of whatever uses them. Things need to be independent