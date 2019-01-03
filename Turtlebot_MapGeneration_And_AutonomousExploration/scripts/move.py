#!/usr/bin/env python


import rospy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from math import radians

counter = 0

def set_globvar():
	global counter
	counter+=1
	return counter

def callback(msg):
	
	if msg.ranges[0]<0.55  or msg.ranges[270]<0.55 or msg.ranges[90]<0.55 or msg.ranges[180]<0.55 or msg.ranges[45]<0.5 or msg.ranges[135]<0.55:
		#while msg.ranges[0]<0.5 or msg.ranges[270]<0.5 or msg.ranges[90]<0.5:		
		print msg.ranges[0]
			
		move_cmd.angular.z = radians(35)
		a = set_globvar()
		print a
		if a < 1 :
			print(a)
			#move_cmd = Twist()
			move_cmd.linear.x = 0.1
			#move_cmd.angular.z = radians(469)
			#move_cmd.angular.z = radians(550)
			
			#move_cmd.angular.z = radians(60)
						
		move_cmd.linear.x = 0.1
		#move_cmd.linear.y = 0.2
	else:
		move_cmd.linear.x = 0.1
		#move_cmd.linear.y = 0.2
		move_cmd.angular.z = 0.0	
			
		

rospy.init_node('obstacle_avoidance')
sub=rospy.Subscriber('/scan',LaserScan, callback)
pub=rospy.Publisher('cmd_vel_mux/input/navi',Twist,queue_size = 10)
r = rospy.Rate(10)
move_cmd = Twist()


while not rospy.is_shutdown():
	pub.publish(move_cmd)
	r.sleep()

rospy.spin()
