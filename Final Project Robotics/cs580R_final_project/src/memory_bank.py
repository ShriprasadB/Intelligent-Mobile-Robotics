#!/usr/bin/env python

"""
memory_bank.py

"""

import os

import numpy as np
import rospy
from rospy_tutorials.msg import Floats
from rospy.numpy_msg import numpy_msg
from std_msgs.msg import String


class MemoryBank(object):
    """External memory bank node."""

    TOLERANCE = 1.  # TODO: test out exact magnitude here

    def __init__(self):

        self.mem_bank = []

        rospy.init_node('mem_bank', anonymous=False)

        rospy.Subscriber('feature_vectors', numpy_msg(Floats),
                         self.feat_vec_callback)

        self.pub = rospy.Publisher('mem_bank_explore', String, queue_size=10)

    def feat_vec_callback(self, feat_vecs):
        """Compares incoming feature vectors to those that are in `mem_bank`.
        Decides where to go because on euclidean distance. Adds these new
        feature vectors to `mem_bank`.
        """
        feat_vecs = feat_vecs.data

        if self.mem_bank:
            l2_dist = self._compute_l2(feat_vecs)
            if l2_dist.min() < self.TOLERANCE:
                self.pub(String("ALREADY EXPLORED"))
            else:
                self.pub(String("ENTER ROOM"))

        self.mem_bank += feat_vecs.tolist()

    def _compute_l2(self, feat_vecs):
        """Computes l2 b/n incoming `feat_vecs` and the feature vectors already
        in `mem_bank`.

        This will result in a 2d array of shape: (# new feature vectors, #
        feature vectors in `mem_bank`). The values will contain the l2 distance
        b/t each new feature vector and every feature vector in `mem_bank`
        (i.e. value for position 2, 8 in array will be the l2 norm b/t the
        third incoming feature vector and the ninth vector in the `mem_bank`).

        Parameters
        ----------
        feat_vecs : numpy.ndarray of shape (# vectors, # features)
            Will contain a number of feature vectors from the doorway of
            a room. Each row will contain a different feature vector.

        Returns
        -------
        l2 : numpy.array
        """
        l2 = np.sqrt(np.square(feat_vecs[:, np.newaxis]
                     - np.asarray(self.mem_bank)).sum(axis=2))
        return l2



if __name__ == '__main__':
    try:
        memory_bank = MemoryBank()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
