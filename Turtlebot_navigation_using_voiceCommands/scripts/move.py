#!/usr/bin/env python

import argparse
import roslib
import rospy
from geometry_msgs.msg import Twist
from pocketsphinx.pocketsphinx import *
from sphinxbase.sphinxbase import *
import pyaudio
from std_msgs.msg import String
from std_srvs.srv import *
import os
import commands

global pub_
 
# initialize ROS
speed = 0.2
msg = Twist()

rospy.init_node('nav')
#rospy.on_shutdown(shutdown)

# you may need to change publisher destination depending on what you run
pub_ = rospy.Publisher("mobile_base/commands/velocity", Twist, queue_size=10)


if rospy.has_param("~model"):
    model = rospy.get_param("~model")
else:
    rospy.loginfo("Loading the default acoustic model")
    model = "/usr/share/pocketsphinx/model/en_US/hub4wsj_sc_8k"
    rospy.loginfo("Done loading the default acoustic model")

if rospy.has_param("~kwlist"):
    kwlist = rospy.get_param("~kwlist")
else:
    rospy.logerr('No dictionary found. Please add an appropriate dictionary argument.')
   

if rospy.has_param("~lexicon"):
    lexicon = rospy.get_param("~lexicon")
else:
    rospy.logerr('kws cant run. Please add an appropriate keyword list file.')
   

# initialize pocketsphinx
config = Decoder.default_config()
config.set_string('-hmm', model)
config.set_string('-dict', lexicon)
config.set_string('-kws', kwlist)

stream = pyaudio.PyAudio().open(format=pyaudio.paInt16, channels=1,rate=16000, input=True, frames_per_buffer=1024)
stream.start_stream()

decoder = Decoder(config)
decoder.start_utt()

while not rospy.is_shutdown():
    buf = stream.read(1024)
    if buf:
          decoder.process_raw(buf, False, False)
    else:
          break
 
    if decoder.hyp() != None:
            print ([(seg.word, seg.prob, seg.start_frame, seg.end_frame)
                 for seg in decoder.seg()])
            print ("Detected keyphrase, restarting search")
            seg.word = seg.word.lower()
            decoder.end_utt()
            decoder.start_utt()
            # you may want to modify the main logic here
            if seg.word.find("bingo") > -1:
     	   	 if speed == 0.2:
        	     msg.linear.x = msg.linear.x*2
        	     msg.angular.z = msg.angular.z*2
        	     speed = 0.4
     	    if seg.word.find("forward") > -1:
        	 msg.linear.x = speed
        	 msg.angular.z = 0
     	    elif seg.word.find("left") > -1:
        	 if msg.linear.x != 0:
        	     if msg.angular.z < speed:
        	         msg.angular.z += 0.05
        	 else:
        	     msg.angular.z = speed*2
     	    elif seg.word.find("right") > -1:
        	 if msg.linear.x != 0:
        	     if msg.angular.z > -speed:
        	         msg.angular.z -= 0.05
        	 else:
        	     msg.angular.z = -speed*2
     	    elif seg.word.find("back") > -1:
        	 msg.linear.x = -speed
        	 msg.angular.z = 0
     	    elif seg.word.find("stop") > -1:
        	 msg = Twist()

    pub_.publish(msg)



