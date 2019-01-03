#!/usr/bin/env python

"""
movement.py

"""
import math
import time
import numpy as np

import rospy
from std_msgs.msg import String
from geometry_msgs.msg import Pose, Point, Quaternion, PoseStamped, Twist
from nav_msgs.msg import Odometry, OccupancyGrid
import actionlib
from actionlib_msgs.msg import GoalStatus
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from rospy_tutorials.msg import Floats
from rospy.numpy_msg import numpy_msg

#http://www.theconstructsim.com/ros-qa-how-to-convert-quaternions-to-euler-angles/
from tf.transformations import euler_from_quaternion, quaternion_from_euler



class Movement(object):

    def __init__(self):

        # TODO: Add ability to set this with command line arguments
        # taget object to find
        self.target_id = 1      # 1 is the id for a person

        # whether to auto explore or not
        self.auto_explore_on = True

        # keep track of approx pose from /odom topic
        self.approx_last_pose = None
        
        rospy.init_node('movement', anonymous=False)

        self.twist_pub = rospy.Publisher('/cmd_vel_mux/input/teleop', Twist,
                                   queue_size=10) 
        self.ac = actionlib.SimpleActionClient('move_base', MoveBaseAction)
        self.ac.wait_for_server(rospy.Duration(5))

        rospy.Subscriber('/odom', Odometry, self.set_approx_pose)
        rospy.Subscriber('processed_detections', numpy_msg(Floats),
                         self.detection_callback)
        rospy.Subscriber('mem_bank_explore', String,
                         self.mem_bank_callback)

    def detection_callback(self, data):
        """Moves toward target object.

        Note
        ----
        FOV of astra camera is 60 degrees.

        TODO
        ----
        > explore this idea:
            https://answers.ros.org/question/203952/move_base-goals-in-base_link-for-turtlebot/
            http://www.theconstructsim.com/ros-qa-how-to-convert-quaternions-to-euler-angles/
            first we query pose to get an idea of where the robot thinks it is;
            then we add our goal to it ... this may work ... but also may not
            if the robot quickly changes its view on where it is right? 
        """
        data = data.data.reshape(-1, 4)

        if self.target_id in set(data[:, 0]):
            self.auto_explore_on = False    # stop auto exploration
            bool_arr = (data[:, 0] == self.target_id)
            target_arr = np.squeeze(data[bool_arr, :])
            width_center, depth = target_arr[[1, 3]]
            deg_rotate = (.5 - width_center) * 60
            rad_rotate = abs(deg_rotate) * math.pi / 180.

            # Two alternatives to movement ... currently actionlib is not
            # working well without a map, even if we use 'base_link'.
            if True:
                if deg_rotate < 0.:
                    rad_rotate *= -1
                move_msg = Twist()
                if depth > 0.4:
                    move_msg.linear.x = .2 * depth
                    move_msg.angular.z = rad_rotate
                self.twist_pub.publish(move_msg)

            if False:
                y_movement_action_lib = depth*math.tan(rad_rotate)
                if deg_rotate < 0.:
                    y_movement_action_lib *= -1
                # gather last position and update
                position = self.approx_last_pose.position
                position.x += depth
                position.y += y_movement_action_lib

                # use last orientation
                orient = self.approx_last_pose.orientation

                # create goal and move towards object
                goal = MoveBaseGoal()
                # goal.target_pose is a PoseStamped msg
                goal.target_pose.header.frame_id = 'map'
                goal.target_pose.header.stamp = rospy.get_rostime()
                goal.target_pose.pose = Pose(position, orient)
                self.ac.send_goal(goal)
                self.ac.wait_for_result(rospy.Duration(60))

                # understand what the GoalStatus object is
                #if ac.get_state() != GoalStatus.SUCCEEDED:
                    # try and move the robot a bit and try again
                    #raise NotImplementedError
    
    def mem_bank_callback(self, data):
        """
        """
        cmd = data.data
        if cmd == "ALREADY EXPLORED":
            # TODO: implement this
            pass
        elif cmd == "ENTER ROOM":
            # TODO: implement this
            pass
        else:
            raise ValueError("topic /mem_bank_explore not understood.")
        
    def _auto_explore(self):
        """
        """
        # TODO: Need to integrate autonomous navigation package here. This is
        #       just a placeholder for now.
        move_msg = Twist()
        move_msg.linear.x = .1
        move_msg.angular.z = .4
        self.twist_pub.publish(move_msg)

    def auto_explore(self):
        """
        """
        rate = rospy.Rate(10)
        while self.auto_explore_on:
            self._auto_explore()
            rate.sleep()

    def set_approx_pose(self, data):
        """
        """
        self.approx_last_pose = data.pose.pose

    def _rotate(self, degrees, direction):
        """Rotates turtlebot.

        Parameters
        ----------
        degrees : int
        direction : str
        """
        move_msg = Twist()
        angle = 0
        t0 = rospy.get_time()

        if direction == 'clockwise':
            move_msg.angular.z = -.4
            rad = -math.pi / (180./degrees)
            while angle > rad:
                self.pub.publish(move_msg)
                t1 = rospy.get_time()
                angle = (t1-t0)*(-.4)
        else:
            move_msg.angular.z = .4
            rad = math.pi / (180./degrees)
            while angle < rad:
                self.pub.publish(move_msg)
                t1 = rospy.get_time()
                angle = (t1-t0)*.4

        # stop turtlebot from rotating due to momentum
        move_msg.angular.z = 0
        self.pub.publish(move_msg)

    

if __name__ == '__main__':
    try:
        movement = Movement()
        movement.auto_explore()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
