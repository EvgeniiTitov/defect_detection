import threading
import os


class ResultsProcessor(threading.Thread):

    def __init__(
            self,
            save_path,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        pass
