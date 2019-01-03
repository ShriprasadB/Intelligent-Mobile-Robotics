#!/usr/bin/env python

"""
test_object_detection.py

This object detection uses the Tensorflow Object Detection API located at:
https://github.com/tensorflow/models/tree/master/research/object_detection

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
from PIL import Image
import rospy

from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
if StrictVersion(tf.__version__) < StrictVersion('1.9.0'):
    err_msg = ('Please upgrade your TensorFlow installation '
               'to v1.9.* or later!')
    raise ImportError(err_msg)


# needed so node can be run from anywhere
SRC_DIR = os.path.dirname(os.path.abspath(__file__))


def load_image_into_numpy_array(image):
    """Converts image to numpy array.

    Parameters
    ----------
    image : PIL image instance

    Returns
    -------
    image_np : numpy.array of type np.uint8

    TODO
    ----
    THIS WILL MOST DEFINITELY NEED TO BE CHANGED WHEN WE FIGURE OUT HOW
    DATA IS GOING TO BE ACQUIRED BY TURTLEBOT.
    """
    (im_width, im_height) = image.size
    return (np.array(image.getdata())
            .reshape((im_height, im_width, 3))
            .astype(np.uint8))

def run_inference_for_single_image(image_np, graph):
    """
    """
    with graph.as_default():
        with tf.Session() as sess:
            # Get handles to input and output tensors
            #print(tf.contrib.graph_editor.get_tensors(tf.get_default_graph()))
            ops = tf.get_default_graph().get_operations()
            for op in ops:
                for output in op.outputs:
                    print "name: " + str(output) + "shape: " + str(output.shape)
            all_tensor_names = {output.name for op in ops for output in op.outputs}
            tensor_dict = {}
            for key in ['num_detections', 'detection_boxes', 'detection_scores',
                        'detection_classes', 'detection_masks']:
                tensor_name = key + ':0'
                if tensor_name in all_tensor_names:
                    tensor_dict[key] = (tf.get_default_graph()
                                        .get_tensor_by_name(tensor_name))

            if 'detection_masks' in tensor_dict:
                # The following processing is only for single image
                detection_boxes = tf.squeeze(tensor_dict['detection_boxes'],
                                             [0])
                detection_masks = tf.squeeze(tensor_dict['detection_masks'],
                                             [0])

                # Reframe is required to translate mask from box coordinates to
                # image coordinates and fit the image size.
                real_num_detection = tf.cast(tensor_dict['num_detections'][0],
                                             tf.int32)
                detection_boxes = tf.slice(detection_boxes, [0, 0],
                                           [real_num_detection, -1])
                detection_masks = tf.slice(detection_masks, [0, 0, 0],
                                           [real_num_detection, -1, -1])

                detection_masks_reframed = \
                    utils_ops.reframe_box_masks_to_image_masks(
                        detection_masks, detection_boxes,
                        image_np.shape[0], image_np.shape[1])
                detection_masks_reframed = tf.cast(
                    tf.greater(detection_masks_reframed, 0.5), tf.uint8)

                # Follow the convention by adding back the batch dimension
                tensor_dict['detection_masks'] = tf.expand_dims(
                    detection_masks_reframed, 0)

            image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')
            # Run inference
            output_dict = sess.run(tensor_dict,
            feed_dict={image_tensor: np.expand_dims(image_np, 0)})
            # all outputs are float32 numpy arrays, so convert types as appropriate
            output_dict['num_detections'] = int(output_dict['num_detections'][0])
            output_dict['detection_classes'] = output_dict[
            'detection_classes'][0].astype(np.uint8)
            output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
            output_dict['detection_scores'] = output_dict['detection_scores'][0]
            if 'detection_masks' in output_dict:
                output_dict['detection_masks'] = output_dict['detection_masks'][0]

    return output_dict


def download_and_extract_model():
    """Downloads and extracts object detection pretrained model.

    Returns
    -------
    frozen_graph_path : str
        Path to the frozen model.
    """
    # Model to download ... look at model zoo for other options.
    download_base = 'http://download.tensorflow.org/models/object_detection/'
    #model_file = 'ssd_mobilenet_v1_coco_2017_11_17.tar.gz'
    model_file = 'faster_rcnn_inception_resnet_v2_atrous_lowproposals_oid_2018_01_28.tar.gz'
    model_path = os.path.join(SRC_DIR, model_file)

    opener = urllib.request.URLopener()
    opener.retrieve(download_base + model_file, model_path)
    tar_file = tarfile.open(model_path)
    for file_ in tar_file.getmembers():
        file_name = os.path.basename(file_.name)
        if file_name == 'frozen_inference_graph.pb':
            tar_file.extract(file_, SRC_DIR)

    # Path to frozen detection graph. This is the actual model that is used for
    # the object detection.
    frozen_graph_path = os.path.join(SRC_DIR,
                                     model_file.partition('.')[0],
                                     'frozen_inference_graph.pb')
    return frozen_graph_path


def load_frozen_graph(frozen_graph_path):
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


def load_labels():
    """Load the label map.

    Returns
    -------
    label_map :
    """
    # List of the strings that is used to add correct label for each box.
    path_to_labels = os.path.join(SRC_DIR, 'labels', 'mscoco_label_map.pbtxt')
    label_map = label_map_util.create_category_index_from_labelmap(
        path_to_labels, use_display_name=True)
    return label_map


def test_object_detection():
    """
    """
    frozen_graph_path = download_and_extract_model()
    detection_graph = load_frozen_graph(frozen_graph_path)
    label_map = load_labels()

    # for now, just test out with a few test images
    test_image_dir = os.path.join(SRC_DIR, 'test_images')
    test_images = [os.path.join(test_image_dir, p)
                   for p in os.listdir(test_image_dir)]

    for image_path in test_images:
        image = Image.open(image_path)
        # the array based representation of the image will be used later in
        # order to prepare the result image with boxes and labels on it.
        image_np = load_image_into_numpy_array(image)
        # Expand dimensions since the model expects images to have shape: [1,
        # None, None, 3]
        image_np_expanded = np.expand_dims(image_np, axis=0)
        # Actual detection.
        output_dict = run_inference_for_single_image(image_np, detection_graph)
        print(output_dict)

        # Visualization of the results of a detection. Assumes we are in a
        # Jupyter Notebook for now ... It doesn't seem to work for python2.7!
        # TODO: how do we use ROS with python3?
        #vis_util.visualize_boxes_and_labels_on_image_array(
        #    image_np,
        #    output_dict['detection_boxes'],
        #    output_dict['detection_classes'],
        #    output_dict['detection_scores'],
        #    label_map,
        #    instance_masks=output_dict.get('detection_masks'),
        #    use_normalized_coordinates=True,
        #    line_thickness=8)


if __name__ == '__main__':
    try:
        time.sleep(.5)  # give other nodes time to initialize
        test_object_detection()
    except rospy.ROSInterruptException:
        pass
