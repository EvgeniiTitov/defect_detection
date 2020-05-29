import cv2
import torch
from .darknet_torch import Darknet
import numpy as np
import sys


class YOLOv3:
    """
    Class for detecting and classifying
    """
    def initialize_model(
            self,
            config,
            weights,
            classes,
            confidence=0.2,
            NMS_threshold=0.2,
            network_resolution=416
    ):
        """
        Initializes network with config and weights provided.
        :param config: path to config
        :param weights: path to weights
        :param classes: path to .txt file with classes listed
        :param confidence:
        :param NMS_threshold:
        :param network_resolution:
        :return:
        """
        # Network's parameters
        self.confidence = confidence
        self.NMS_threshold = NMS_threshold
        self.network_resolution = network_resolution
        self.batch_size = 1

        self.classes = self.load_classes(classes)
        self.num_classes = len(self.classes)

        # Load model using cfg and weights provided
        self.model = Darknet(config)
        self.model.load_weights(weights)

        self.input_dimension = int(self.model.net_info["height"])

        # Check CUDA availability. Push model into GPU memory if available
        self.CUDA = torch.cuda.is_available()

        if self.CUDA:
            self.model.cuda()

        # Set model to .eval() so that we do not change its parameters during
        # testing
        self.model.eval()

    def predict(self, image):
        """
        Detects objects on the image provided
        :param image: list of np.ndarrays
        :return:
        """
        img = self.preprocess_image(img=image)

        im_dim = image.shape[1], image.shape[0]
        im_dim = torch.FloatTensor(im_dim).repeat(1, 2)

        if self.CUDA:
            img = img.cuda()
            im_dim = im_dim.cuda()
            # print("Moved to CUDA")

        image = img.squeeze(0).permute(1, 2, 0)
        image = image.cpu().numpy()
        import os
        save_path = r"D:\Desktop\system_output\INPUT\1\result"
        cv2.imwrite(os.path.join(save_path, "master_before_net.jpg"), image)

        with torch.no_grad():
            # Row bounding boxes are predicted. Note predictions from from
            # 3 YOLO layers get concatenated into 1 big tensor.
            raw_predictions = self.model(img, self.CUDA)

        # BBs need to be filtered by object confidence and NMS
        # tensor of shape [Nb of objects found, 8]
        # 8: 4 BBs coordinates, objectness score, max conf score and its index
        output = self.process_predictions(raw_predictions)

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

    def process_predictions(self, predictions):
        """
        Filter out BBs by 1) Thresholding by object confidence, 2) NMS
        :param predictions:
        :return: Tensor of shape D x 8.
        - index of the image in the batch - 0
        - 4 BBs coordinates
        - objectness score
        - the score of class with max confidence
        - index of this class
        """

        # predictions.shape torch.Size([1, 22743, 7]), where 1 is batch size,
        # 4 BBs coordinates + 1 objectness score + 2 classes for metal/concrete
        # 22743 / 2 - Nb of BBs predicted per class
        # print(predictions.shape)  # len(predictions[0]) = 22743

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

        # Conf thresholding and NMS need to be done for each image in the batch
        # separately
        for ind in range(batch_size):
            image_pred = predictions[ind]  # image Tensor
            # confidence threshholding
            # NMS

            max_conf, max_conf_score = torch.max(image_pred[:, 5:5 + self.num_classes], 1)
            max_conf = max_conf.float().unsqueeze(1)
            max_conf_score = max_conf_score.float().unsqueeze(1)
            seq = (image_pred[:, :5], max_conf, max_conf_score)
            image_pred = torch.cat(seq, 1)

            non_zero_ind = (torch.nonzero(image_pred[:, 4]))
            try:
                image_pred_ = image_pred[non_zero_ind.squeeze(), :].view(-1, 7)
            except:
                continue

            if image_pred_.shape[0] == 0:
                continue
            #

            # Get the various classes detected in the image
            img_classes = self.unique(image_pred_[:, -1])  # -1 index holds the class index

            for cls in img_classes:
                # perform NMS

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

                batch_ind = image_pred_class.new(image_pred_class.size(0), 1).fill_(
                    ind)  # Repeat the batch_id for as many detections of the class cls in the image
                seq = batch_ind, image_pred_class

                if not write:
                    output = torch.cat(seq, 1)
                    write = True
                else:
                    out = torch.cat(seq, 1)
                    output = torch.cat((output, out))

        # Check if anything has been detected at all
        try:
            return output
        except:
            return 0

    def unique(self, tensor):
        """

        :param tensor:
        :return:
        """
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
        # Resize image to match the network resolution
        img = (self.letterbox_image(img, (self.input_dimension, self.input_dimension)))
        # RGB -> Something
        img = img[:, :, ::-1].transpose((2, 0, 1)).copy()
        # Create torch object, normalize all pixels, add extra dimension?
        img = torch.from_numpy(img).float().div(255.0).unsqueeze(0)

        return img

    def letterbox_image(self, img, inp_dim):
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
