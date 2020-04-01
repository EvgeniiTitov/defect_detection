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
- You pass frame between workers whereas its already stored inside each detected pole object
- BBs modification. Overlapping check.
- Class names and accuracy once written look bad
- 2 pillars can potentially get generated
- Not well optimized line filtering algorithm


-------------------CURRENT THINGS TO DO-----------------------------
1. Rewrite threads in accordance with Garlic's advices

    1.1 Endpoint predict needs to return video ID.

    1.2 Combine photo and video pipelines, for both respond with ID

    1.3 Cleaning progress dictionary - add status field success. If true -> delete from dict
        Or when results saved - delete object from the dictionary

    1.4 Separate pipeline for inclination calculations.
        Create a separate worker - similar to object detector. You will just
        have a bit of logic what worker to choose! Each can work with various
        models (for some case all 3 classes, for some just 1)

2. Test memory consumption to tune Qs
3. Add method to join threads, shut down the whole server
    3.1 Where does it called from? - Separate endpoint?


ASK OLEG:
1. Can we do polling to display progress?  or websockets or one request at once?
Server can timeout if it takes too long to process a video
We could try to extend timeout time though
2. Polling or sockets can be useful to display processing progress OR websocket that can
once in a while report back the progress and then results

Endpoint predict - puts item(s) in the Q and returns ID(s) (maybe save path)
Endpoint status - allows to track progress