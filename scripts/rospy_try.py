#!/usr/bin/env python 2.7
import rospy
import message_filters
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
import cv2
from cv_bridge import CvBridge, CvBridgeError
from pose_estimation import *

rospy.loginfo("Starts")
bridge=CvBridge()


#def createWindow():
#	cv2.namedWindow('Image Window')

def show_image(img):
	cv2.imshow("Image Window",img)
	cv2.waitkey(3)

def image_callback(img,pose):
	#rospy.loginfo(img.header)
	rospy.loginfo(pose.pose.position)
	try:
		cv_image=bridge.imgmsg_to_cv2(img,"passthrough")
	except CvBridgeError, e:
		rospy.logerr("CvBridge Error: {0}".format(e))
	#Pose_estimate(cv_image)	
	RGB_img = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
	Pose_estimate(RGB_img)
	#cv2.waitKey(1)


#def pose_callback(pose):
#	rospy.loginfo(pose.pose.position.x)
#sub_image=rospy.Subscriber("/downward_cam/camera/image",Image,image_callback)
#sub_pose=rospy.Subscriber("/ground_truth_to_tf/pose",PoseStamped,pose_callback)

#createWindow()

sub_image=message_filters.Subscriber("/downward_cam/camera/image",Image)
sub_pose=message_filters.Subscriber("/ground_truth_to_tf/pose",PoseStamped)
sync=message_filters.ApproximateTimeSynchronizer([sub_image,sub_pose],10,3)
sync.registerCallback(image_callback)



while not rospy.is_shutdown():
	rospy.init_node('info')
	rospy.spin()	

