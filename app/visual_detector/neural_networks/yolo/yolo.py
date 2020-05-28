import cv2
import torch
from .darknet_torch import Darknet
from torch.autograd import Variable
from torchvision import transforms
import torch.nn.functional as F
from typing import List, Dict
import numpy as np
import torchvision
import sys
from app.visual_detector.utils import HostDeviceManager


'''
Step 1 - Tower detection
You need to load original images on GPU. 
Then for YOLO, resize them, do predictions, recalculate bb coordinates for original sizes
Return original size images on GPU and coordinates on them 

Step 2 - Component detection
Gets images on GPU + tower coordinates. Crops images on GPU, resizes them, ,cat() in a batch,
get component detections, recalculate their bb coordinates for original sizes
Returns

Step 3 - Defect detection
Same

TODO:
c) If you have N images in a batch, M towers on them, you will get Z components predicted. 
   How to match predicted components with the towers they belong to? 
'''


class YOLOv3:
    def __init__(
            self,
            config,
            weights,
            classes,
            confidence=0.2,
            NMS_threshold=0.2,
            network_resolution=416
    ):
        # Network's parameters
        self.confidence = confidence
        self.NMS_threshold = NMS_threshold
        self.network_resolution = network_resolution
        self.batch_size = 1
        self.classes = self.load_classes(classes)
        self.num_classes = len(self.classes)

        # Load model using cfg and weights provided
        try:
            self.model = Darknet(config)
            self.model.load_weights(weights)
        except Exception as e:
            print(f"Failed during YOLO initialization. Error: {e}")
            raise

        # Determines size of an input image required
        self.input_dimension = int(self.model.net_info["height"])

        # Check CUDA availability. Push model into GPU memory if available
        self.is_model_on_gpu = False
        self.CUDA = torch.cuda.is_available()
        if self.CUDA:
            try:
                self.model.cuda()
                self.is_model_on_gpu = True
            except Exception as e:
                print(f"Failed to move model to GPU. Error: {e}")

        # Set model to .eval() so that we do not change its parameters during testing
        self.model.eval()

    def process_batch(self, images: torch.Tensor) -> dict:
        """
        Receives a batch of images already preprocessed and loaded onto the GPU. Simply runs the net and post-
        processes the predictions
        :param images:
        :return:
        """
        assert isinstance(images, torch.Tensor), "The batch provided is of the wrong data type. Torch tensor expected"
        assert images.is_cuda == self.is_model_on_gpu, "The provided batch and model on different devices"

        # Make a copy and resize images to the expected size keeping the aspect ratio
        # TODO: Copy your oriignal images, they must not be resized
        copy_images = images
        resized_images = HostDeviceManager.resize_tensor(tensor=copy_images, new_size=self.input_dimension)
        HostDeviceManager.visualise_sliced_img(resized_images)
        print("IS CUDS:", resized_images.is_cuda)

        # Run the batch of images through the net
        with torch.no_grad():
            # Row bounding boxes are predicted. Note predictions from
            # 3 YOLO layers get concatenated into 1 big tensor.
            raw_predictions = self.model(Variable(resized_images), self.CUDA)

        # Process raw predictions by filtering out results using NMS and thresholding
        output = self._process_predictions(raw_predictions)

        # Recalculate bb coordinates relatively to the images of the original size
        # TODO: Recalculate BB relatively to the original image

        return output

    def predict_batch_on_cpu(self, images: List[np.ndarray]) -> tuple:
        """
        Receives a list of images. Preprocesses them (cast to torch.Tensor, reshape etc), .cat()s them in a batch
        and them runs it throught the net.
        :param images:
        :return:
        """
        # Prepare images - resize, move to torch.Tensor, .cat() them together
        preprocessed_imgs = list(map(self.preprocess_image, images))
        try:
            batch = torch.cat(preprocessed_imgs)
        except Exception as e:
            print(f"Failed during .cat() ing images. Error: {e}")
            raise

        # Save dimensions of the original images
        dimensions_list = [(img.shape[1], img.shape[0]) for img in images]
        dimensions_list = torch.FloatTensor(dimensions_list).repeat(1, 2)

        if self.CUDA:
            dimensions_list = dimensions_list.cuda()
            batch = batch.cuda()

        with torch.no_grad():
            # Row bounding boxes are predicted. Note predictions from
            # 3 YOLO layers get concatenated into 1 big tensor.
            raw_predictions = self.model(Variable(batch), self.CUDA)

        # Process raw predictions, do NMS, thresholding etc
        output = self._process_predictions(raw_predictions)

        return output, batch

    def predict(self, image: np.ndarray) -> list:
        """
        Runs the net on only 1 image
        :param image: list of np.ndarrays
        :return:
        """
        img = self.preprocess_image(img=image)

        im_dim = image.shape[1], image.shape[0]
        im_dim = torch.FloatTensor(im_dim).repeat(1, 2)

        if self.CUDA:
            img = img.cuda()
            im_dim = im_dim.cuda()

        with torch.no_grad():
            # Row bounding boxes are predicted. Note predictions from
            # 3 YOLO layers get concatenated into 1 big tensor.
            raw_predictions = self.model(img, self.CUDA)

        # BBs need to be filtered by object confidence and NMS
        # tensor of shape [Nb of objects found, 8]
        # 8: 4 BBs coordinates, objectness score, max conf score and its index
        output = self._process_predictions(raw_predictions)

        # Got 0 instead of Tensor object. Nothing's been detected
        if type(output) == int:
            return

        # UNFOLDING REDUCED (RESIZED) IMAGE. CONFIRM!
        im_dim = im_dim.repeat(output.size(0), 1)
        scaling_factor = torch.min(self.network_resolution / im_dim, 1)[0].view(-1, 1)

        output[:, [1, 3]] -= (self.input_dimension - scaling_factor * im_dim[:, 0].view(-1, 1)) / 2
        output[:, [2, 4]] -= (self.input_dimension - scaling_factor * im_dim[:, 1].view(-1, 1)) / 2

        output[:, 1:5] /= scaling_factor

        for i in range(output.shape[0]):
            output[i, [1, 3]] = torch.clamp(output[i, [1, 3]], 0.0, im_dim[i, 0])
            output[i, [2, 4]] = torch.clamp(output[i, [2, 4]], 0.0, im_dim[i, 1])

        return output.tolist()

    def _process_predictions(self, predictions: torch.Tensor) -> Dict[int, list]:
        """
        :param predictions:
        :return:
        """
        '''
        Output (predictions) tensor shape: batch_size, 10647, 5 + nb_of_classes. The output
        needs to be filtered by a) Thresholding by object confidence, b) NMS
        
        Object Confidence Thresholding
        Prediction tensor contain info about batch_size x 10647 bbs. For all bbs whose conf 
        threshold < confidence, set all its values to 0.
        '''
        # Each BBs with objectness score < threshold, get all its values set to 0
        conf_mask = (predictions[:, :, 4] > self.confidence).float().unsqueeze(2)
        predictions = predictions * conf_mask

        # Its easier to calculate IoU of two boxes using coordinates of a pair of
        # diagonal corners of each box. So, we transform (centre x, centre y, H, W)
        # to (top-left X, top-left Y, right-bot X, right-bot Y)
        box_corner = predictions.new(predictions.shape)
        box_corner[:, :, 0] = (predictions[:, :, 0] - predictions[:, :, 2] / 2)
        box_corner[:, :, 1] = (predictions[:, :, 1] - predictions[:, :, 3] / 2)
        box_corner[:, :, 2] = (predictions[:, :, 0] + predictions[:, :, 2] / 2)
        box_corner[:, :, 3] = (predictions[:, :, 1] + predictions[:, :, 3] / 2)
        predictions[:, :, :4] = box_corner[:, :, :4]

        # Batch size - nb of images
        batch_size = predictions.size(0)
        write = False

        # Save predictions for each image in the batch
        output_results = {i: [] for i in range(batch_size)}
        # Conf thresholding and NMS need to be done for each image in the batch separately
        for ind in range(batch_size):
            image_pred = predictions[ind]  # image Tensor

            # Clean up
            max_conf, max_conf_score = torch.max(image_pred[:, 5:5 + self.num_classes], 1)
            max_conf = max_conf.float().unsqueeze(1)
            max_conf_score = max_conf_score.float().unsqueeze(1)
            seq = (image_pred[:, :5], max_conf, max_conf_score)
            image_pred = torch.cat(seq, 1)

            # Get rid of rows whose confidence was < the conf thresh, so we set their rows to 0s
            non_zero_ind = (torch.nonzero(image_pred[:, 4]))
            try:
                image_pred_ = image_pred[non_zero_ind.squeeze(), :].view(-1, 7)
            except:
                continue
            if image_pred_.shape[0] == 0:
                continue

            # Get the various classes detected in the image
            img_classes = self.unique(image_pred_[:, -1])  # -1 index holds the class index

            # perform NMS
            for cls in img_classes:
                # get the detections with one particular class
                cls_mask = image_pred_ * (image_pred_[:, -1] == cls).float().unsqueeze(1)
                class_mask_ind = torch.nonzero(cls_mask[:, -2]).squeeze()
                image_pred_class = image_pred_[class_mask_ind].view(-1, 7)

                # sort the detections such that the entry with the maximum objectness
                # confidence is at the top
                conf_sort_index = torch.sort(image_pred_class[:, 4], descending=True)[1]
                image_pred_class = image_pred_class[conf_sort_index]
                idx = image_pred_class.size(0)  # Number of detections

                for i in range(idx):
                    # Get the IOUs of all boxes that come after the one we are looking at
                    # in the loop
                    try:
                        ious = self.bbox_iou(image_pred_class[i].unsqueeze(0), image_pred_class[i + 1:])
                    except ValueError:
                        break
                    except IndexError:
                        break

                    # Zero out all the detections that have IoU > treshhold
                    iou_mask = (ious < self.NMS_threshold).float().unsqueeze(1)
                    image_pred_class[i + 1:] *= iou_mask

                    # Remove the non-zero entries
                    non_zero_ind = torch.nonzero(image_pred_class[:, 4]).squeeze()
                    image_pred_class = image_pred_class[non_zero_ind].view(-1, 7)

                # # Repeat the batch_id for as many detections of the class cls in the image
                # batch_ind = image_pred_class.new(image_pred_class.size(0), 1).fill_(ind)
                # seq = batch_ind, image_pred_class

                output_results[ind].append(image_pred_class.tolist())

        return output_results

    def unique(self, tensor):
        """
        :param tensor:
        :return:
        """
        # TODO: Look into this .cpu() transfer. This is ridiculous
        tensor_np = tensor.cpu().numpy()
        unique_np = np.unique(tensor_np)
        unique_tensor = torch.from_numpy(unique_np)
        tensor_res = tensor.new(unique_tensor.shape)
        tensor_res.copy_(unique_tensor)

        return tensor_res

    def bbox_iou(self, box1, box2):
        """
        Returns the IoU of two bounding boxes


        """
        # Get the coordinates of bounding boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

        # get the corrdinates of the intersection rectangle
        inter_rect_x1 = torch.max(b1_x1, b2_x1)
        inter_rect_y1 = torch.max(b1_y1, b2_y1)
        inter_rect_x2 = torch.min(b1_x2, b2_x2)
        inter_rect_y2 = torch.min(b1_y2, b2_y2)

        # Intersection area
        inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * torch.clamp(
            inter_rect_y2 - inter_rect_y1 + 1,
            min=0)

        # Union Area
        b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
        b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

        iou = inter_area / (b1_area + b2_area - inter_area)

        return iou

    def load_classes(self, filename):
        """
        Loads and returns classes
        :param filename:
        :return:
        """
        with open(filename, "r") as file:
            names = file.read().split("\n")

        return names

    def preprocess_image(self, img):
        """
        Prepare image for inputting to the neural network.
        Transforms numpy object to PyTorch's input format (changes the order to PuyTorch's
        one: BatchSize x Channels x Height x Width
        """
        # Resize image to match the network resolution. For instance: (416, 416, 3)
        img = (self.letterbox_image(img, (self.input_dimension, self.input_dimension)))
        # Swap colour axis. Result: (3, 416, 416)
        img = img[:, :, ::-1].transpose((2, 0, 1)).copy()
        # Create torch object, normalize all pixels, add extra dimension - batch size at the beginning
        # Result: torch.Size([1, 3, 416, 416])
        img = torch.from_numpy(img).float().div(255.0).unsqueeze(0)

        return img

    def letterbox_image(self, img, inp_dim: tuple):
        """
        Resize image with unchanged aspect ratio using padding. Left out areas after
        resizing get painted in 128,128,128 colour.
        :param img:
        :param inp_dim:
        :return:
        """
        img_w, img_h = img.shape[1], img.shape[0]
        w, h = inp_dim
        new_w = int(img_w * min(w / img_w, h / img_h))
        new_h = int(img_h * min(w / img_w, h / img_h))
        resized_image = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        canvas = np.full((inp_dim[1], inp_dim[0], 3), 128)
        canvas[(h - new_h) // 2:(h - new_h) // 2 + new_h, (w - new_w) // 2:(w - new_w) // 2 + new_w, :] = resized_image

        return canvas
