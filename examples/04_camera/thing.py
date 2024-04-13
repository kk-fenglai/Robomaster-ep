#!usr/bin/python#xcoding:utf-x#义编码，中文注释
#import the necessary packagas
import nunpy as np
import Cv2
# 找到国标医
def find_marker(image):
# canvert the inage to grayscele, blur it, and detect edge
    gray = cv2.cvtColor(imag, cv2.COLOR PGR2GRAY)
    gray=cv2.GaussianBlur(gray,(5,5),0)
    edged=cv2.canny(gray,35,125)
r find the contours in the cdged inagc and kecp the largest onc;# we'1l assume that this is our piece of paper in the imege(cnts,)c2.findContours(edged,COpy(),CV2 RETR LIST,CV2.CHAIN APPROX SIHPLE)# 求品大百积
cnax(cnts,keycv2.contaurArea)
# conpute the bounding bax of the of the paper regiun and return it#cv2.ninareaRect()c代表点集，返同rec[0]是最小外接距形中心点坐标，#rect16]主width,rect[11[1]是he1gnt，rect[2]是角黛return cv2.minAreaRectic)
，距高计賀的数def distance to_camera(knonnwidth, focalLergth, perwidth):# comgute and return the distance fron the maker to the caneraretumm (knounlldth*focallength) perwidth
# initialize the known distance fron the canera to the cbject, which# in this case is 24 inchesKHOMH DISTAMCE  24.8
# initialize the known ohjact width, which in this case, tha pfece af# paper is 1l inches wide#A4新的长和宽(单位:inches)
KMOWN NIDTH- 11.69KMOMN HEIGHT = 8,27
# initialize the list af imagos that wo'1l be usingIMAGE_PAIHS -["Picture1.jpg", "picture2.jpg”, "picture3.jpg”]
# load the furst inage that contains an obfect that is KNON TO DE 2 feet#from our camera, then find the paper marker in the image, and initial1zt the facal length#读入第一张图，通过已知距高计算相机焦距inegecv2.inread(IMWE _PATHS[B])narker  find markercinage)focalLength = (marKer[1][6]* KHOW DISTANCE) ' KHOMN_WIDTH
#雨过播像头标主洗股的像素篇距#focalLength - 811.82print('focalLength",focalLength)
时打]开播像头
canora  cv2.widaocspture(a)
while canera.isopened():# pet a frane(grabbed,frane)* canera.read()narcar = find marker(frane)if narker.. 8:print(marker)continueinches= distance_to_canera《KNOw IDTH, focalLength, narker[1][0])
# drau a bounding box around the image and displey itbox  np.int0(cv2.cv.BoxPoints(marke})cw2.drawcontours(frame,[box]，-1，(0，255，0，2)
# inches 转换为 ccv2.putText(frum,"%.2fcm”%(inches *38.49/ 12)。(frane shape[1]-200,frame,shape[0]- 20), CV2 FONT HERSHEY SIMPLEX,2.6,(0，255，6)，3)
t hw a franocv2 inshow("captur!", frane)if cv2.maitKey(1)8 0xFF ord( q”):breakcanera.release()cv2.dostroyAllwindaws()