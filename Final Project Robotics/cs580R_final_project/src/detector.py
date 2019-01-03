#!/usr/bin/env python

"""
detector.py

This object detection uses the Tensorflow Object Detection API located at:
https://github.com/tensorflow/models/tree/master/research/object_detection

Some ideas were inspired by the following repositories:
    > https://github.com/osrf/tensorflow_object_detector
    > https://github.com/cagbal/ros_people_object_detection_tensorflow

TODO
----
> Need to figure out which feature vectors to use.

Feature vectors taken from the following network architecture:

ssd_mobilenet_v1 --> trained on COCO

Here are some options for ssd mobilenet:
    name:
    Tensor("FeatureExtractor/MobilenetV1/Conv2d_13_pointwise_1_Conv2d_2_1x1_256/Relu6:0",
    shape=(?, 10, 10, 256), dtype=float32)

    name:
    Tensor("FeatureExtractor/MobilenetV1/Conv2d_13_pointwise_1_Conv2d_3_1x1_128/Relu6:0",
    shape=(?, 5, 5, 128), dtype=float32)

    name:
    Tensor("FeatureExtractor/MobilenetV1/Conv2d_13_pointwise_1_Conv2d_4_1x1_128/Relu6:0",
    shape=(?, 3, 3, 128), dtype=float32)

    name:
    Tensor("FeatureExtractor/MobilenetV1/Conv2d_13_pointwise_2_Conv2d_5_3x3_s2_128/Relu6:0",
    shape=(?, 1, 1, 128), dtype=float32)

"""
import os
import six.moves.urllib as urllib
import sys
import tarfile
import zipfile
from distutils.version import StrictVersion
from collections import defaultdict
from io import StringIO
import time

import numpy as np
import tensorflow as tf
import rospy
from rospy_tutorials.msg import Floats
from rospy.numpy_msg import numpy_msg
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2D
import cv2
from cv_bridge import CvBridge, CvBridgeError

from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

if StrictVersion(tf.__version__) < StrictVersion('1.9.0'):
    err_msg = ('Please upgrade your TensorFlow installation '
               'to v1.9.* or later!')
    raise ImportError(err_msg)


# needed so that node can be run from anywhere
SRC_DIR = os.path.dirname(os.path.abspath(__file__))


class ObjectDetector(object):
    """Object detection node.

    Published Topics
    ----------------
    '/processed_detections'
    '/feature_vectors'

    Subscribed Topics
    -----------------
    '/camera/depth/image_raw'
    '/camera/rgb/image_raw'

    """
    def __init__(self):

        # parameters ... allows customization by user
        self._model_param = rospy.get_param('/_model_name')
        self._depth_correction = float(rospy.get_param('/_depth_correction'))
        self._confidence_cutoff = float(rospy.get_param('/_confidence_cutoff'))
        print str(type(self._confidence_cutoff))

        # tensorflow model
        self.detection_graph = self._init_detection_network()
        self.tensor_dict = self._init_tensor_handles()
        self.label_map = self._load_labels()
        self.sess = self._init_sess()

        # opencv interface
        self.bridge = CvBridge()

        rospy.init_node('detector', anonymous=False)

        # just hold last available depth frame to approx. depth of
        # current rgb frame
        rospy.Subscriber('/camera/depth/image_raw', Image,
                         self.process_depth_frame,
                         queue_size=1, buff_size=2**24)
        self.last_depth_frame = None

        rospy.Subscriber('/camera/rgb/image_raw', Image,
                         self.run_inference_for_single_image,
                         queue_size=1, buff_size=2**24)

        # TODO: Whichever node receives this message needs to reshape it as
        #       follows: np_array.reshape(-1, 4).
        self.obj_detect_pub = rospy.Publisher('processed_detections',
                                              numpy_msg(Floats),
                                              queue_size=10)

        self.feat_vec_pub = rospy.Publisher('feature_vectors',
                                            numpy_msg(Floats),
                                            queue_size=10)

    def _download_and_extract_model(self):
        """Downloads and extracts object detection pretrained model.

        Returns
        -------
        frozen_graph_path : str
            Path to the frozen model.
        """
        poss_model_files = {
            'ssd_mobilenet_v1': 'ssd_mobilenet_v1_coco_2017_11_17.tar.gz',
            'faster_rcnn_inception_resnet_v2': (
                'faster_rcnn_inception_resnet_v2_atrous_lowproposals_oid_'
                '2018_01_28.tar.gz'),
        }
        if self._model_param not in poss_model_files:
            raise ValueError('Name of model_name parameter not correct')

        # Model to download ... look at model zoo for other options.
        download_base = 'http://download.tensorflow.org/models/object_detection/'
        model_file = poss_model_files[self._model_param]
        model_path = os.path.join(SRC_DIR, 'models', model_file)
        frozen_graph_path = os.path.join(SRC_DIR, 'models',
                                         model_file.partition('.')[0],
                                         'frozen_inference_graph.pb')

        # need to check if exists first b/c `exist_ok` arg does not exist
        # for `makedirs` in python2.7
        if not os.path.exists(frozen_graph_path):
            os.makedirs(os.path.dirname(model_path))
            opener = urllib.request.URLopener()
            opener.retrieve(download_base + model_file, model_path)
            tar_file = tarfile.open(model_path)
            for file_ in tar_file.getmembers():
                file_name = os.path.basename(file_.name)
                if file_name == 'frozen_inference_graph.pb':
                    tar_file.extract(file_, os.path.join(SRC_DIR, 'models'))

        return frozen_graph_path

    def _load_frozen_graph(self, frozen_graph_path):
        """Loads frozen graph into memory.

        Parameters
        ----------
        frozen_graph_path : str
            Path to the frozen graph.

        Returns
        -------
        detection_graph : tf.Graph instance
            Represents the frozen graph with all pretrained weights.
        """
        detection_graph = tf.Graph()
        with detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(frozen_graph_path, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
        return detection_graph

    def _init_detection_network(self):
        """Initialize the object detection graph

        Returns
        -------
        detection_graph : tf.Graph instance
            Represents the frozen graph with all pretrained weights.
        """
        path = self._download_and_extract_model()
        detection_graph = self._load_frozen_graph(path)
        return detection_graph

    def _init_tensor_handles(self):
        """Initializes a handle to important tensors in graph.

        Returns
        -------

        """
        assert hasattr(self, 'detection_graph')

        # Get handles to tensors
        tensor_names = [
            'num_detections',
            'detection_boxes',
            'detection_scores',
            'detection_classes',
        ]
        # add feature vector
        if self._model_param == 'ssd_mobilenet_v1':
            tensor_names.append(
                'FeatureExtractor/MobilenetV1/Conv2d_13_pointwise_2_Conv2d_5_'
                '3x3_s2_128/Relu6')
        elif self_model_param == 'faster_rcnn_inception_resnet_v2':
            raise NotImplementedError("Need to look up feat vector for resnet.")
        else:
            raise ValueError

        tensor_dict = {
            name: self.detection_graph.get_tensor_by_name(name + ':0')
            for name in tensor_names
        }

        # change name of feature vector to make it more readable
        if self._model_param == 'ssd_mobilenet_v1':
            tensor_dict['feature_vectors'] = tensor_dict.pop(
                'FeatureExtractor/MobilenetV1/Conv2d_13_pointwise_2_Conv2d_5_'
                '3x3_s2_128/Relu6')
        elif self_model_param == 'faster_rcnn_inception_resnet_v2':
            raise NotImplementedError("Need to look up feat vector for resnet.")
        else:
            raise ValueError

        return tensor_dict

    def _load_labels(self):
        """Load the label map.

        Returns
        -------
        label_map :
        """
        # List of the strings that is used to add correct label for each box.
        if self._model_param == 'ssd_mobilenet_v1':
            path_to_labels = os.path.join(SRC_DIR, 'labels',
                                          'mscoco_label_map.pbtxt')
        elif self_model_param == 'faster_rcnn_inception_resnet_v2':
            path_to_labels == os.path.join(SRC_DIR, 'labels',
                                           'oid_bbox_trainable_label_map.pbtxt')
        else:
            raise ValueError

        label_map = label_map_util.create_category_index_from_labelmap(
            path_to_labels, use_display_name=True)

        return label_map

    def _init_sess(self):
        """Initializes a tensorflow session.

        Returns
        -------
        tf.Session
        """
        assert hasattr(self, 'detection_graph')
        return tf.Session(graph=self.detection_graph)

    def _convert_rgb_to_np_array(self, data):
        """Converts RGB sensor_msgs/Image data to numpy array."""
        cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        return np.asarray(image)

    def _convert_depth_to_np_array(self, data):
        """Converts depth sensor_msgs/Image data to numpy array."""
        cv_image = self.bridge.imgmsg_to_cv2(data, "32FC1")
        return np.asarray(cv_image)

    def run_inference_for_single_image(self, data):
        """Run object detection for incoming image.

        Callback for the rospy.Subscriber('/camera/rgb/image_raw')

        Decscription of information returned from network:

        num_detections : int
            Number of returned detections. Will always be 100 for
            ssd_mobilenet_v1.
        detection_classes : numpy.ndarray
            Contains class ids for detected objects. Will have
            shape=(`num_detections`, ).
        detection_boxes : numpy.ndarray
            Contains 4 coordinates of bounding box for each object in
            `detection_classes`. Coordinates are in the order: ymin, xmin,
            ymax, xmax. Coordinates are normalized b/t 0.0 and 1.0.  Has
            shape=(`num_detections`, 4).
        detection_scores : numpy.ndarray
            Confidence of classification for each object in `detection_classes`.
            Has shape=(`num_detections`, ).
        'feature_vectors': np.array
            Contains abstract representation of scene. Just a feature vector
            taken from a higher layer in the network.

        """
        while self.last_depth_frame is None:
            time.sleep(1)

        image_np = self._convert_rgb_to_np_array(data)
        depth_frame_np = self.last_depth_frame
        assert image_np.shape[:2] == depth_frame_np.shape[:2]
        height, width, _ = image_np.shape

        image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')

        # Run inference
        output_dict = self.sess.run(
            self.tensor_dict,
            feed_dict={image_tensor: np.expand_dims(image_np, 0)})

        # all outputs are float32 numpy arrays, so convert types as appropriate
        output_dict['num_detections'] = int(output_dict['num_detections'][0])
        output_dict['detection_classes'] = (output_dict['detection_classes'][0]
                                            .astype(np.uint8))
        output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
        output_dict['detection_scores'] = output_dict['detection_scores'][0]
        output_dict['feature_vectors'] = (output_dict['feature_vectors'][0]
                                         .flatten())

        # publish feature vectors
        feature_vectors = output_dict.pop('feature_vectors')
        self.feat_vec_pub.publish(feature_vectors)

        # publish information summary from detection network
        # TODO: Either figure out how to send multi dimensional numpy
        #       arrays or just reshape upon send/receive of this topic.
        processed_detections = self.process_detections(depth_frame_np, width,
                                                       height, **output_dict)
        if processed_detections is not None:
            self.obj_detect_pub.publish(processed_detections.flatten())


    def process_detections(self, depth_frame_np, width, height, num_detections,
                           detection_classes, detection_boxes,
                           detection_scores):
        """Process information obtained from object detection network.

        Information about the 100 most confident objects is returned from
        the object detection network. Often, the confidence on many of these
        objects is very small. We only want to look at the objects that
        the network is confident about.

        We will return the center point of the most confident bounding
        box for each unique object (provided that the confidence meets
        a certain threshold).

        Parameters
        ----------
        depth_frame_np : numpy.ndarray
            Numpy frame of depth image.
        width : int
            The width (in pixels) of RGB/Depth images taken by astra camera.
        height : int
            The height (in pixels) of RGB/Depth images taken by astra camera.
        num_detections : int
            Number of returned detections. Will always be 100 for
            ssd_mobilenet_v1.
        detection_classes : numpy.ndarray w/ dtype=uint8
            Contains class ids for detected objects. Will have
            shape=(`num_detections`, ).
        detection_boxes : numpy.ndarray w/ dtype float
            Contains 4 coordinates of bounding box for each object in
            `detection_classes`. Coordinates are in the order: ymin, xmin,
            ymax, xmax. Coordinates are normalized b/t 0.0 and 1.0.  Has
            shape=(`num_detections`, 4).
        detection_scores : numpy.ndarray w/ dtype float
            Confidence of classification for each object in `detection_classes`.
            Has shape=(`num_detections`, ).

        Returns
        -------
        numpy.ndarray w/ dtype np.float32
            Has shape (# of confident detections, 4). By 'confident', we mean
            the number of detections we deem to be confident enough to
            consider.  Will contain the object id, the normalized center x
            coordinate, the normalized center y coordinate, and the approximate
            depth of the object in meters.  If detection network was not
            confident about any object, then returns None.
        """
        # trim to only consider boxes with high confidence
        tr_scores = detection_scores[detection_scores > self._confidence_cutoff]
        tr_num = tr_scores.shape[0]
        tr_classes = detection_classes[:tr_num]
        tr_boxes = detection_boxes[:tr_num, :]

        # if nothing confident exists, return None
        if tr_num == 0:
            return None

        # trim to only consider most confident object from each class
        uq, idx = np.unique(tr_classes, return_index=True)
        idx = np.sort(idx)

        tr_scores = tr_scores[idx]
        tr_classes = tr_classes[idx]
        tr_boxes = tr_boxes[idx]

        centers_width = (tr_boxes[:, 1] + tr_boxes[:, 3]) / 2
        centers_height = (tr_boxes[:, 0] + tr_boxes[:, 2]) / 2
        depth_medians = self._median_depth_of_boxes(depth_frame_np, width,
                                                    height, tr_boxes)

        to_concat = [
            np.expand_dims(tr_classes, 1),      # class id
            np.expand_dims(centers_width, 1),   # normalized center pixel x
            np.expand_dims(centers_height, 1),  # normalized center pixel y
            np.expand_dims(depth_medians, 1),   # approx depth of object
        ]

        objects = np.concatenate(to_concat, axis=1).astype(np.float32)

        return objects

    def process_depth_frame(self, data):
        """Saves the last depth frame as numpy array.

        Callback for the rospy.Subscriber('/camera/depth/image_raw')
        """
        self.last_depth_frame = self._convert_depth_to_np_array(data)
        # Apply depth correction. For some reason the astra camera returns
        # depth in mm, whereas gazebo returns depth in meters.
        self.last_depth_frame *= self._depth_correction

    def _median_depth_of_boxes(self, depth_frame_np, width, height, boxes):
        """Approximates objects depth by using median depth on bounding box.

        Parameters
        ----------
        depth_frame_np : np.array
            Numpy frame of depth image.
        width : int
            The width (in pixels) of RGB/Depth images taken by astra camera.
        height : int
            The height (in pixels) of RGB/Depth images taken by astra camera.
        boxes : np.array of shape (num boxes, 4)
            Each row is a box. The 4 columns are ymin, xmin, ymax, xmax

        Returns
        -------
        np.array
            Approximate depth (meters) of the object in each of the bounding
            boxes.
        """
        pixel_boxes = boxes * np.array([height, width, height, width])
        pixel_boxes = pixel_boxes.astype(int)

        def _depth_median(arr):
            """Excludes np.nan's and zeros from median depth calculation."""
            return np.median(arr[~np.logical_or(np.isnan(arr), arr == 0.)])

        approx_depth = np.array([
            _depth_median(depth_frame_np[box[0]:box[2], box[1]:box[3]])
            for box in pixel_boxes
        ])

        return approx_depth


if __name__ == '__main__':
    try:
        detector = ObjectDetector()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
