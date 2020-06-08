import threading



class DumperClassifierThread(threading.Thread):

    def __init__(
            self,
            in_queue,
            out_queue,
            dumper_classifier,
            progress,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.Q_in = in_queue
        self.Q_out = out_queue
        self.dumper_classifier = dumper_classifier
        self.progress = progress

    def run(self) -> None:

        while True:

            input_ = self.Q_in.get()
            if input_ == "STOP":
                break

            #print("Dumpers:", input_)

            # Collect sliced out tensors and send to the model for preproccessing and
            # subsequent inference
            images_to_preprocess = list()
            for obj, obj_tensor in input_:
                images_to_preprocess.append(obj_tensor)

            # Get results from the dumper classifier
            output = self.dumper_classifier.predict(images_to_preprocess)

            # Modify dumper objects state accordingly
            assert len(output) == len(images_to_preprocess), "Nb of images and nb of predictions do not match"
            for i in range(len(output)):
                input_[i][0].deficiency_status = True if output[i] == "defected" else False

            self.Q_out.put("Success")

        print("DumperClassifierThread killed")
