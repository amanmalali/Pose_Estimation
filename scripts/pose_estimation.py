import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import glob, os
import sys
from math import sqrt,cos,sin

#sys.path.insert(1, '/home/sadgun/RBCCPS/MBZ/Code/')
#from Utils import utility
class markers():
    def __init__(self,color,position,rotation,name):
        self.color=color
        self.position=position
        self.rotation=rotation
        self.name=name


class GetPose:
    '''
    TODO: Add the description of the class and its members here. 
    '''
    #  TODO Init the tiny yolo model in the constuctor of the class and test the same 


    # 3D model points.
    # model_points = np.array([
    #     [0.0, 0.0, 0.0],
    #     [0.0, 160.0, 0.0],
    #     [-20.0, 160.0, 0.0],
    #     [-20.0, -20.0, 0.0],
    #     [150.0, -20.0, 0.0],
    #     [150.0, 0.0, 0.0]

    # ])
    # model_points = np.array([
    #     [0.0, 0.0, 0.0],
    #     [0.0, 160.0, 0.0],
    #     [-20.0, 160.0, 0.0],
    #     [-20.0, -20.0, 0.0],
    #     [120.0, -20.0, 0.0],
    #     [120.0, 0.0, 0.0]

    # ])
    # model_points = np.array([
    #     [0.0, 0.0, 0.0],
    #     [0.0, 150.0, 0.0],
    #     [-105.0, 150.0, 0.0],
    #     [-105.0, -105.0, 0.0],
    #     [300.0, -105.0, 0.0],
    #     [300.0, 0.0, 0.0]

    # ])
    model_points = np.array([
        [0.0, 0.0, 0.0],
        [0.0, 300.0, 0.0],
        [-105.0, 300.0, 0.0],
        [-105.0, -105.0, 0.0],
        [150.0, -105.0, 0.0],
        [150.0, 0.0, 0.0]])
    #f = (width/2) / tan(hfov/2)

    camera_matrix = np.array(
        [[268.511, 0.000000, 332.706],
         [0.000000, 268.511, 240.841],
         [0.000000, 0.000000, 1.000000]], dtype="double"
    )

    dist_coeffs = np.array([[0.0],
                            [0.0],
                            [0.0],
                            [0.0],
                            [0.0]])
    #[76,177,34],[2045,1745],90,"green"
    def init_color_markers(self):
        yellow_marker=markers([0,255,255],[-640,350],0,"yellow")
        black_marker=markers([0,0,0],[-1292,1108],0,"black")
        green_marker=markers([76,177,34],[545,-245],1.5708,"green")
        blue_marker=markers([232,162,0],[1290,-1108],3.14159,"blue")
        red_marker=markers([36,28,237],[1105,1288],4.71239,"red")
        purple_marker=markers([164,73,163],[-1097,-1288],1.5708,"purple")
        marker_array=[yellow_marker,blue_marker,black_marker,green_marker,purple_marker,red_marker]
        return marker_array

    def calc_distance(self,marker,image_obj):
        return (sqrt(sum((np.array(marker)-np.array(image_obj))**2)))

    def translate_axis(self,marker,position):
        x=position[0]*cos(marker.rotation)-position[1]*sin(marker.rotation)
        y=position[0]*sin(marker.rotation)+position[1]*cos(marker.rotation)
        # x=marker.position[0]+position[0]
        # y=marker.position[1]+position[1]
        print("Rotation: ",x,y)
        x_rot=marker.position[0]+x
        y_rot=marker.position[1]+y
        print("Translation: ",x_rot,y_rot)
        return [x_rot,y_rot,position[2]]
    def findHist_YUV(self, rgbimage):

        yuv = cv.cvtColor(rgbimage, cv.COLOR_BGR2YUV)
        y, u, v = yuv[:, :, 0], yuv[:, :, 1], yuv[:, :, 2]

        #utility.showSaveImg(v)

        hist_y = cv.calcHist([yuv], [0], None, [256], [0, 256])
        hist_u = cv.calcHist([yuv], [1], None, [256], [0, 256])
        hist_v = cv.calcHist([yuv], [2], None, [256], [0, 256])
        print("b argmax: {}".format(np.argmax(hist_y, axis=0)))
        print("g argmax: {}".format(np.argmax(hist_u, axis=0)))
        print("r argmax: {}".format(np.argmax(hist_v, axis=0)))

        return v,y, np.argmax(hist_v, axis=0)

    def find_freq(self, rgbimage,mask):

        rgb = rgbimage
        b, g, r = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]

        #utility.showSaveImg(v)

        hist_b = cv.calcHist([rgb], [0], mask, [256], [0, 256])
        hist_g = cv.calcHist([rgb], [1], mask, [256], [0, 256])
        hist_r = cv.calcHist([rgb], [2], mask, [256], [0, 256])
        # print("b argmax: {}".format(np.argmax(hist_b, axis=0)))
        # print("g argmax: {}".format(np.argmax(hist_g, axis=0)))
        # print("r argmax: {}".format(np.argmax(hist_r, axis=0)))

        return np.array([np.argmax(hist_b, axis=0),np.argmax(hist_g, axis=0),np.argmax(hist_r, axis=0)])


    def thresholdHistImg(self, hsvImg,yhsvImg,histIdx):

        #  TODO make this part more dynamic 
        threshImg = cv.inRange(hsvImg, histIdx-5 , histIdx+5)
        threshImg_black = cv.inRange(yhsvImg,0,5)
        threshImg=cv.bitwise_or(255-threshImg,threshImg_black)
        
        #utility.showimg(threshImg)
        return 255-threshImg

    def getCornerPoints(self, contour):

        flag = True
        delta = 0.01
        it=0
        while (flag):
            it+=1
            epsilon = delta * cv.arcLength(contour, True)
            approx = cv.approxPolyDP(contour, epsilon, True)
            approx = approx.squeeze()
            if(it==5):
                return approx,-1
            if len(approx) != 6:
                delta += 0.01
                flag = True
            else:
                flag = False
                return approx,1
        

    def getPose(self, rgbimage, rgbPath=""):
        marker_info=self.init_color_markers()
        vImg,yImg,hist_idx = self.findHist_YUV((rgbimage))
        threshImg = self.thresholdHistImg(vImg,yImg,hist_idx)

        kernel = cv.getStructuringElement(cv.MORPH_ERODE, (7, 7))
        eroded = cv.erode(threshImg, kernel, iterations=4)
        kernel = cv.getStructuringElement(cv.MORPH_DILATE, (5, 5))
        dilated = cv.dilate(eroded, kernel, iterations=4)
        #utility.showimg(dilated)

        dilated = 255 - threshImg
        #cv.waitKey(2)
        #utility.showimg(dilated)

        # get the contours of the segmented image
        _, contours, _ = cv.findContours(dilated, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)  # find contours

        image_clone = rgbimage
        img = cv.drawContours(image_clone, contours, -1, (0, 255, 0), 1)
        #utility.showimg(img)
        # cv.imshow('Image Window',img)
        # cv.waitKey(0)
        cntsSorted = sorted(contours, key=lambda x: cv.contourArea(x), reverse=True)
        if(len(contours)==0):
            print("crashes")
            return -1,-1,-1
        #print("len of contours: {}".format(len(contours)))
        #cnt = cntsSorted[0]
        num_markers_visible=0
        avg_position=np.zeros(3)
        for cnt in cntsSorted:
            if(cv.contourArea(cnt)<1000):
                continue
            #print("Area : ",cv.contourArea(cnt))
            mask = np.zeros(rgbimage.shape[:2], np.uint8)
            #poly=cv.approxPolyDP(cnt,0.1,True)
            cv.drawContours(mask, [cnt], -1, 255, -1)
            mean=self.find_freq(rgbimage,mask).squeeze()
            #cv.waitKey(3)
            
            #print(mean)
            min_dist=1000
            for mark in marker_info:
                dist=self.calc_distance(mark.color,mean)
                #print(mark.name," Distance : ",dist)
                if(dist<min_dist):
                    min_dist=dist
                    color=mark
            # get the corner points from the contours, we compute this in a loop such that we always return 6 corner points
            approx,flag = self.getCornerPoints(cnt)
            if(flag==-1):
                return -1,-1,-1
            #print("approx \n {}".format(approx))

            # compute the convexity defects the identify the concave points and start labelling the corners in clockwise direction
            hull = cv.convexHull(approx, returnPoints=False, clockwise=True)
            defects = cv.convexityDefects(approx, hull)
            # print('len def ', len(defects))
            if(defects==None):
                return -1,-1,-1
            if len(defects) > 1:
                defects = defects[len(defects) - 1]
            defects = defects.squeeze()
            c_idx = defects[2]
            # shift_idx = len(approx)-(c_idx+1)
            shift_idx = len(approx) - (c_idx)
            ordered = np.roll(approx, shift_idx, axis=0)


            points_img = rgbimage
            
            point = tuple(approx[defects[2]])
            # print(point)
            # for cirpoints in approx:
            #     cirpoints = tuple(cirpoints)
            #     cv.circle(points_img, cirpoints, 4, [0, 255, 0], -1)
            # cv.circle(points_img, point, 4, [0, 0, 255], -1)
            # cv.imshow('Image Window',img)
            # cv.waitKey(0)
            # utility.showimg(points_img)

            # clone = rgbimage
            # cv.putText(clone,'P 1',tuple(ordered[0]), 1, 1,(0,255,0),2,cv.LINE_AA)
            # cv.putText(clone,'P 2',tuple(ordered[1]), 1, 1,(0,255,0),2,cv.LINE_AA)
            # cv.putText(clone,'P 3',tuple(ordered[2]), 1, 1,(0,255,0),2,cv.LINE_AA)
            # cv.putText(clone,'P 4',tuple(ordered[3]), 1, 1,(0,255,0),2,cv.LINE_AA)
            # cv.putText(clone,'P 5',tuple(ordered[4]), 1, 1,(0,255,0),2,cv.LINE_AA)
            # cv.putText(clone,'P 6',tuple(ordered[5]), 1, 1,(0,255,0),2,cv.LINE_AA)
            # utility.showimg(clone)

            image_points = ordered.astype(float)
            success, rotation_vector, translation_vector = cv.solvePnP(self.model_points, image_points, self.camera_matrix,
                                                                      self.dist_coeffs)
        

            # Re-project the points and plot the axis
            #axis = np.float32([[50,0,0], [0,50,0], [0,0,-30]]).reshape(-1,3)
            #imgpts, jac = cv.projectPoints(axis, rotation_vector, translation_vector, self.camera_matrix, self.dist_coeffs)
            
            # origImg = cv.imread(rgbPath)
            # img = utility.draw(origImg,ordered.astype(int),imgpts)
            # utility.showimg(img)
            
            rotM = cv.Rodrigues(rotation_vector)[0]
            cameraPosition = -np.matrix(rotM).T * np.matrix(translation_vector)
            #print("Camera position : ")
            #print(cameraPosition)
            cameraPosition=np.array(cameraPosition.squeeze())
            world_coord=self.translate_axis(color,cameraPosition[0])
            num_markers_visible+=1
            avg_position=avg_position+np.array(world_coord)
            print("Position wrt marker :",num_markers_visible)
            print("Localizing with : ",color.name)
            print("Camera position :",cameraPosition)
            print("World coordinates: ",world_coord)
            
            #print("len: ",len(marker_info))
        print("Average position : ",avg_position/num_markers_visible)
        return -1,-1,-1
        #return success, np.degrees(rotation_vector), translation_vector 


        #return approx, point, ordered


def Pose_estimate(image):
    
    # imagePath = "/home/aman/hector_quad/images/frame000642.jpg"
    # image = cv.imread(imagePath)

    #utility.showimg(image)

    pose = GetPose()
    success, rotation_vector, translation_vector  = pose.getPose(image)
    #if(success!=-1):
    #print("rotation vector: {}".format(rotation_vector))
    #print("translation vector: {}".format(translation_vector))

    # folderPath = "/home/sadgun/RBCCPS/MBZ/Dataset/Challenge_2/data_19th_sept/out_images/"

    # filepath = "/home/sadgun/RBCCPS/Docs/Sep30_demo_video/"
    # out = cv.VideoWriter(filepath+"result.avi", cv.VideoWriter_fourcc(*"MJPG"), 15.0,(1280, 720))
    # for imgnum in range(1879, 2190):
    #     imagePath = folderPath+"rgb_image_"+str(imgnum)+".jpg"
    #     image = cv.imread(imagePath)
    #     pose = GetPose()
    #     approx, point, ordered, imgpts = pose.getPose(image)

    #     for cirpoints in approx:
    #         cirpoints = tuple(cirpoints)
    #         cv.circle(image, cirpoints, 4, [0, 255, 0], -1)
    #     cv.circle(image, point, 4, [0, 0, 255], -1)

    #     img = utility.draw(image, ordered.astype(int),imgpts)
    #     out.write(img)
    
