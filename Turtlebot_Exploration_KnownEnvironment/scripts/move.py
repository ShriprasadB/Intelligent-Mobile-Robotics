#!/usr/bin/env python

import rospy
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
import actionlib
from actionlib_msgs.msg import *
from geometry_msgs.msg import Pose, Point, Quaternion

goalsent = False

#co-ordinates of points on a map
xx = [0.1,-3.83,-1.54,3]
yy = [-3.1,-5.28,-2.28,-5.8]

def shutdown():
        if goalsent:
            motion.cancel_goal()
        rospy.loginfo("Stop")
        rospy.sleep(1)


rospy.init_node('nav', anonymous=False)
rospy.on_shutdown(shutdown)
motion = actionlib.SimpleActionClient("move_base", MoveBaseAction)
rospy.loginfo("action server wait")
# wait time for action server 
motion.wait_for_server(rospy.Duration(5))

for i in range(0,4):
	# give the respective co-ordinates in a map
	coordinate = {'x': xx[i], 'y' : yy[i]}
	orient = {'r1' : 0.000, 'r2' : 0.000, 'r3' : 0.000, 'r4' : 1.000}
	rospy.loginfo("Go to (%s, %s) pose", coordinate['x'], coordinate['y'])
        # to send a respective goal
        goalsent = True
	goal = MoveBaseGoal()
	goal.target_pose.header.frame_id = 'map'
	goal.target_pose.header.stamp = rospy.Time.now()
        goal.target_pose.pose = Pose(Point(coordinate['x'], coordinate['y'], 0.000),
                                     Quaternion(orient['r1'], orient['r2'], orient['r3'], orient['r4']))
	# start navigation
        motion.send_goal(goal)
	# 100 seconds for the turtlebot to conquer a job
	success = motion.wait_for_result(rospy.Duration(100)) 
        state = motion.get_state()
        result = False
        if success and state == GoalStatus.SUCCEEDED:
            result = True
        else:
            motion.cancel_goal()

        goalsent = False
        success = result
	if success:
		rospy.loginfo("bingoo!! co-ordinate")
	else:
		rospy.loginfo("we are on the correct path")

	rospy.sleep(1)        


