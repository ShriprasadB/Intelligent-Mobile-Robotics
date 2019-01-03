#!/usr/bin/env python

import rospy
from turtlesim.msg import *
from turtlesim.srv import *
from geometry_msgs.msg import Twist
from std_srvs.srv import *
import random
from math import atan2,sqrt, pow

#spawn a turtle
def toSpawnRunner():
    global runnerx, runnery
    runnerx = random.randint(1, 9)
    runnery = random.randint(1, 9)
    spawnTurtle(runnerx,runnery,0,"runner")
         
#to check the distance and kill turtle
def distanceChecking():
    global motion
    # calculates distance between runner and hunter
    distance = sqrt(pow((runnerx-turtle1x),2) + pow((runnery-turtle1y),2))
    runnerTheta = atan2(runnery - turtle1y, runnerx - turtle1x)
    if distance >= 1:
	motion.linear.x = 1
    	motion.angular.z =  (runnerTheta - turtle1theta)
    elif distance < 1:
	killTurtle("runner")
	toSpawnRunner()
    pub.publish(motion)
    
#this will give hunter's position
def hunterPose(data):
    global turtle1x, turtle1y, turtle1theta
    turtle1x = data.x
    turtle1y = data.y
    turtle1theta = data.theta

#this will give runner's position
def runnerPose(data):
    global runnerx, runnery
    runnerx = data.x
    runnery = data.y
        
#to make publisher and subscriber and call other methods
if __name__ == '__main__':
    try:
        global pub,pub1, rate, motion,turtle1x, turtle1y, runnerx, runnery
        rospy.init_node('turtleHunt', anonymous=True)
	pub = rospy.Publisher('/turtle1/cmd_vel', Twist, queue_size = 10)
        pub1 = rospy.Publisher('/runner/cmd_vel', Twist, queue_size = 10)
	#Getting the hunter's Pose
        rospy.Subscriber("/turtle1/pose", Pose, hunterPose)
	#Getting the runner's Pose 
        rospy.Subscriber("/runner/pose", Pose, runnerPose) 
	#The rate of our publishing is 2 seconds Hence 1/2 = 0.5 hertz
        rate = rospy.Rate(0.5) 
	#to make stage empty
        clearStage = rospy.ServiceProxy('/clear', Empty) 
	#spawning
        spawnTurtle = rospy.ServiceProxy('/spawn', Spawn)
	#to kill a turtle 
        killTurtle = rospy.ServiceProxy('/kill', Kill) 
        motion = Twist() 
	motion1 = Twist()
	try:
        	killTurtle("runner")
    	except:
        	pass
    	clearStage()
    	toSpawnRunner()
 	#this will run until we don't give shutdown command
    	while not rospy.is_shutdown():
		motion1.linear.x = 1
		motion1.linear.y = 0
		motion1.linear.z = 0
		motion1.angular.z = random.randint(-1,1)
		pub1.publish(motion1)
		distanceChecking()

    except rospy.ROSInterruptException:
        pass
