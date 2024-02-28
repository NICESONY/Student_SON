#!/usr/bin/env python
# -*- coding: utf-8 -*- 10
# 자이카 C모델 Ubuntu 20.04 + ROS Noetica
#=============================================
#=============================================
# 함께 사용되는 각종 파이썬 패키지들의 import 선언부
#=============================================
import numpy as np
import cv2, rospy, time, math
from sensor_msgs.msg import Image
from std_msgs.msg import Int32MultiArray
from xycar_msgs.msg import xycar_motor
from cv_bridge import CvBridge
from ar_track_alvar_msgs.msg import AlvarMarkers
from tflite_runtime.interpreter import Interpreter
import importlib.util

#=============================================
# 프로그램에서 사용할 변수, 저장공간 선언부
#=============================================
motor = None  # 모터 노드 변수
Fix_Speed = 28 # 모터 속도 고정 상수값 
new_angle = 0  # 모터 조향각 초기값
new_speed = Fix_Speed  # 모터 속도 초기값
bridge = CvBridge()  # OpenCV 함수를 사용하기 위한 브릿지 
ultra_msg = None  # 초음파 데이터를 담을 변수
ultra_data = None  # 초음파 토픽의 필터링에 사용할 변수
image = np.empty(shape=[0])  # 카메라 이미지를 담을 변수
motor_msg = xycar_motor()  # 카메라 토픽 메시지
WIDTH, HEIGHT = 640, 480  # 카메라 이미지 가로x세로 크기
Blue =  (255,0,0) # 파란색
Green = (0,255,0) # 녹색
Red =   (0,0,255) # 빨간색
Yellow = (0,255,255) # 노란색
stopline_num = 1 # 정지선 발견때마다 1씩 증가
View_Center = WIDTH//2  # 화면의 중앙값 = 카메라 위치
ar_msg = {"ID":[],"DX":[],"DZ":[]}  # AR태그 토픽을 담을 변수
interpreter = None  # 객체인식 머신러닝 모델 객체

#=============================================
# 학습 결과물 파일의 위치와 파일이름 지정
#=============================================
PATH_TO_CKPT = '/home/pi/xycar_ws/src/study/colab_tflite/src/detect.tflite'
PATH_TO_LABELS = '/home/pi/xycar_ws/src/study/colab_tflite/src/labelmap.txt'

#=============================================
# 차선 인식 프로그램에서 사용할 상수 선언부
#=============================================
CAM_FPS = 30  # 카메라 FPS 초당 30장의 사진을 보냄
WIDTH, HEIGHT = 640, 480  # 카메라 이미지 가로x세로 크기
ROI_START_ROW = 300  # 차선을 찾을 ROI 영역의 시작 Row값
ROI_END_ROW = 380  # 차선을 찾을 ROT 영역의 끝 Row값
ROI_HEIGHT = ROI_END_ROW - ROI_START_ROW  # ROI 영역의 세로 크기  
L_ROW = 40  # 차선의 위치를 찾기 위한 ROI 안에서의 기준 Row값 

#=============================================
# 프로그램에서 사용할 이동평균필터 클래스
#=============================================
class MovingAverage:

    def __init__(self, n):
        self.samples = n
        self.data = []
        self.weights = list(range(1, n + 1))

    def add_sample(self, new_sample):
        if len(self.data) < self.samples:
            self.data.append(new_sample)
        else:
            self.data = self.data[1:] + [new_sample]
            
    def get_sample_count(self):
        return len(self.data)
        
    # 이동평균값을 구하는 함수
    def get_mavg(self):
        return float(sum(self.data)) / len(self.data)

    # 중앙값을 사용해서 이동평균값을 구하는 함수
    def get_mmed(self):
        return float(np.median(self.data))

    # 가중치를 적용하여 이동평균값을 구하는 함수        
    def get_wmavg(self):
        s = 0
        for i, x in enumerate(self.data):
            s += x * self.weights[i]
        return float(s) / sum(self.weights[:len(self.data)])
        
#=============================================
# 초음파 8개의 거리정보에 대해서 이동평균필터를 적용하기 위한 선언
#=============================================
avg_count = 5  # 이동평균값을 계산할 데이터 묶음 갯수    
ultra_mvavg = [MovingAverage(avg_count) for i in range(8)]

#=============================================
# 콜백함수 - USB 전방카메라 토픽을 받아서 처리하는 콜백함수.
#=============================================
def usbcam_callback(data):
    global image
    image = bridge.imgmsg_to_cv2(data, "bgr8")

#=============================================
# 콜백함수 - 초음파 토픽을 받아서 처리하는 콜백함수.
#=============================================
def ultra_callback(data):
    global ultra_msg, ultra_data
    ultra_data = data.data

    # 이동평균필터를 적용해서 튀는 값을 제거해서 ultra_msg에 담기
    for i in range(8):
        ultra_mvavg[i].add_sample(float(ultra_data[i]))
    
	# 여기서는 중앙값(Median)을 이용해서 튀는 값 제거 - 평균값 또는 가중평균값을 사용하는 것도 가능    
    ultra_list = [int(ultra_mvavg[i].get_wmavg()) for i in range(8)]
    ultra_msg = tuple(ultra_list)




#=============================================
# 콜백함수 - AR태그 토픽을 받아서 처리하는 콜백함수.
#=============================================
def ar_callback(data):
    global ar_msg

    # AR태그의 ID값, X 위치값, Z 위치값을 담을 빈 리스트 준비
    ar_msg["ID"] = []
    ar_msg["DX"] = []
    ar_msg["DZ"] = []

    # 발견된 모두 AR태그에 대해서 정보 수집하여 ar_msg 리스트에 담음
    for i in data.markers:
        ar_msg["ID"].append(i.id) # AR태그의 ID값을 리스트에 추가
        ar_msg["DX"].append(i.pose.pose.position.x) # AR태그의 X 위치값을 리스트에 추가
        ar_msg["DZ"].append(i.pose.pose.position.z) # AR태그의 Z 위치값을 리스트에 추가
    
#=============================================
# 모터 토픽을 발행하는 함수.  
#=============================================
def drive(angle, speed):
    motor_msg.angle = angle
    motor_msg.speed = speed
    motor.publish(motor_msg)
    
#=============================================
# 차량을 정차시키는 함수.  
# 입력으로 시간(초)를 받아 그 시간동안 속도=0 토픽을 모터로 보냄
#=============================================
def stop_car(sleep_sec):
    for i in range(sleep_sec*5): 
        drive(angle=new_angle, speed=0)
        time.sleep(0.1)
    
#=============================================
# 초음파 센서를 이용해서 벽까지의 거리를 알아내서
# 벽과 충돌하지 않으며 주행하도록 핸들 조정함.
#=============================================
def sonic_drive():
    global new_angle, new_speed

    # 앞쪽 가까이에 장애물이 있으면 차량 멈춤
    if (0 < ultra_msg[2] < 3):
        new_angle = new_angle
        new_speed = 0
        print("Car Brake, Stop! : ", ultra_msg)

    elif (0 < ultra_msg[2] < 60  and ultra_msg[3]-ultra_msg[1] > 10):
        new_angle = 95
        new_speed = 15
        print("Car Brake, Stop! : ", ultra_msg)

    # 왼쪽이 오른쪽보다 멀리 있으면 있으면 좌회전 주행
    elif (ultra_msg[1]-ultra_msg[3] > 10):
        new_angle = -40
        new_speed = Fix_Speed
        print("Turn left1 : ", ultra_msg)

        
    # 오른쪽이 왼쪽보다 멀리 있으면 있으면 우회전 주행
    elif (ultra_msg[3]-ultra_msg[1] > 10):
        new_angle = 40
        new_speed = Fix_Speed
        print("Turn left1 : ", ultra_msg)


    # 위 조건에 해당하지 않는 경우라면 (오른쪽과 왼쪽이 동일한 경우) 똑바로 직진 주행
    else:
        new_angle = 0
        new_speed = Fix_Speed
        print("Go Straight : ", ultra_msg)

    # 모터에 주행명령 토픽을 보낸다
    drive(new_angle, new_speed)

#=============================================
# 카메라 이미지를 영상처리하여 
# 정지선이 있는지 체크하고 True 또는 False 값을 반환.
#=============================================
def check_stopline():
    global stopline_num

    # 원본 영상을 화면에 표시
    #cv2.imshow("Original Image", image)
    
    # image(원본이미지)의 특정영역(ROI Area)을 잘라내기
    roi_img = image[250:480, 0:640]
    cv2.imshow("ROI Image", roi_img)


    hsv_image = cv2.cvtColor(roi_img, cv2.COLOR_BGR2HSV) 
    upper_white = np.array([255, 255, 255])
    lower_white = np.array([0, 0, 235])   ###
    binary_img = cv2.inRange(hsv_image, lower_white, upper_white)
    cv2.imshow("Black&White Binary Image", binary_img)  ####



    # 흑백이진화 이미지에서 특정영역을 잘라내서 정지선 체크용 이미지로 만들기
    stopline_check_img = binary_img[100:130, 150:480]  ###
    cv2.imshow("Stopline Check Image", stopline_check_img)  ###


    
    # 흑백이진화 이미지를 칼라이미지로 바꾸고 정지선 체크용 이미지 영역을 녹색사각형으로 표시
    img = cv2.cvtColor(binary_img, cv2.COLOR_GRAY2BGR)
    cv2.rectangle(img, (150,100),(480,130),Green,3)  ###
    cv2.imshow('Stopline Check', img)
    cv2.waitKey(1)

    
    # 정지선 체크용 이미지에서 흰색 점의 개수 카운트하기
    stopline_count = cv2.countNonZero(stopline_check_img)


    
    # 사각형 안의 흰색 점이 기준치 이상이면 정지선을 발견한 것으로 한다
    if stopline_count > 2200:  
        print("Stopline Found...! -", stopline_num)
        stopline_num = stopline_num + 1
        cv2.destroyWindow("ROI Image")
        return True
    
    else:
        return False
    
    
#=============================================
# 카메라 이미지를 영상처리하여 
# 신호등의 출발 신호를 체크해서 True 또는 False 값을 반환.
#=============================================
def check_traffic_sign():
    
    # 원본이미지를 복제한 후에 특정영역(ROI Area)을 잘라내기
    cimg = image.copy()
    Center_X, Center_Y = 320, 100  # ROI 영역의 중심위치 좌표 
    XX, YY = 60, 45  # 위 중심위치 좌표에서 좌우로 XX만큼씩, 상하로 YY만큼씩 벌려서 ROI 영역을 잘라냄   

    # ROI 영역을 녹색 사각형으로 그려서 화면에 표시함 
    cv2.rectangle(cimg, (Center_X-XX, Center_Y-YY), (Center_X+XX, Center_Y+YY) , Green, 2)
	
	# 원본 이미지에서 ROI 영역만큼 잘라서 roi_img에 담음 
    roi_img = cimg[Center_Y-YY:Center_Y+YY, Center_X-XX:Center_X+XX]

    # 칼라 이미지를 회색 이미지로 바꿈  
    img = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)

    # 노이즈 제거를 위해 블러링 처리를 함 
    blur = cv2.GaussianBlur(img, (5, 5), 0)

    # Hough Circle 변환을 이용해서 이미지에서 원을 (여러개) 찾음 
    circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, 1, 20,
                  param1=40, param2=20, minRadius=15, maxRadius=25)

    # 디버깅을 위해서 Canny 처리를 했을때의 모습을 화면에 표시함
	# 위 HoughCircles에서 param1, param2에 사용했던 값을 아래 canny에서 똑같이 적용해야 함. 순서 조심.
    canny = cv2.Canny(blur, 20, 40)   ## 여기 순서 위에 반영하는 코드임#####중요하다고 함###
    cv2.imshow('Area for HoughCircles', canny)

    # 이미지에서 원이 발견됐다면 아래 if 안으로 들어가서 신호등 찾는 작업을 진행함  
    if circles is not None:

        circles = np.round(circles[0, :]).astype("int")

        # 원 중심의 X좌표값으로 소팅 - 화면의 왼쪽부터 순서대로 다시 정리 
        circles = sorted(circles, key=lambda circle: circle[0])

        # 가장 밝은 원을 찾을 때 사용할 변수 선언 
        max_mean_value = 0
        max_mean_value_circle = None
        max_mean_value_index = None

        # 발견된 원들에 대해서 루프를 돌면서 하나씩 처리 
 	    # 원의 중심좌표, 반지름. 내부밝기 정보를 구해서 화면에 출력  
        for i, (x, y, r) in enumerate(circles):
            x1 = x - r // 2
            y1 = y - r // 2
            x2 = x + r // 2
            y2 = y + r // 2
            roi = img[y1:y2, x1:x2]
            mean_value = np.mean(roi)
            print(f"Circle {i} at ({x},{y}), radius={r}: mean value={mean_value}")
			
            # 여기에 발견된 원들 중에서 가장 밝은 원을 찾는 코드가 추가되어야 함 
             # Check if the current circle has a higher mean value
            if mean_value > max_mean_value:
                max_mean_value = mean_value
                max_mean_value_circle = (x, y, r)   ######## 내가 추가한 코드임 
                max_mean_value_index = i
            
            # 찾은 원을 녹색으로 그리고, 원 안에 작은 빨간색 사각형(밝기 정보를 계산할 영역 표시)을 그림 
            cv2.circle(cimg, (x+Center_X-XX, y+Center_Y-YY), r, Green, 2)
            cv2.rectangle(cimg, (x1+Center_X-XX, y1+Center_Y-YY), (x2+Center_X-XX, y2+Center_Y-YY), Red, 2)

        # 가장 밝은 원을 찾았으면 그 원의 정보를 화면에 출력 
        if max_mean_value_circle is not None:
            (x, y, r) = max_mean_value_circle
            print(f" --- Circle {max_mean_value_index} has the biggest mean value")

        # 신호등 찾기 결과가 그림으로 표시된 이미지를 화면에 출력
        cv2.imshow('Circles Detected', cimg)
        cv2.waitKey(1000)
    
	    # 찾은 원 중에서 오른쪽 3번째 원이 가장 밝으면 (파란색 신호등) True 리턴 
        if (i == 2) and (max_mean_value_index == 2):
            print("Traffic Sign is Blue...!")
            cv2.destroyWindow('Area for HoughCircles')
            return True
        
		# 그렇지 않으면 (파란색 신호등이 아니면) False 반환 
        else:
            print("Traffic Sign is NOT Blue...!")
            return False

    # 원본 이미지에서 원이 발견되지 않았다면 False 리턴   
    print("Can't find Traffic Sign...!")
    return False

#=============================================
# 카메라 영상 이미지에서 차선을 찾아 그 위치를 반환하는 코드
#=============================================
def lane_detect():

    global image
    prev_x_left = 0
    prev_x_right = WIDTH

    img = image.copy() # 이미지처리를 위한 카메라 원본이미지 저장
    display_img = img  # 디버깅을 위한 디스플레이용 이미지 저장
    
    # img(원본이미지)의 특정영역(ROI Area)을 잘라내기
    roi_img = img[ROI_START_ROW:ROI_END_ROW, 0:WIDTH]
    line_draw_img = roi_img.copy()

    #=========================================
    # 원본 칼라이미지를 그레이 회색톤 이미지로 변환하고 
    # 블러링 처리를 통해 노이즈를 제거한 후에 (약간 뿌옇게, 부드럽게)
    # Canny 변환을 통해 외곽선 이미지로 만들기
    #=========================================
    gray = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)
    blur_gray = cv2.GaussianBlur(gray,(5, 5), 0)
    edge_img = cv2.Canny(np.uint8(blur_gray), 60, 75)

    # 잘라낸 이미지에서 HoughLinesP 함수를 사용하여 선분들을 찾음
    all_lines = cv2.HoughLinesP(edge_img, 1, math.pi/180,50,50,20)
    
    if all_lines is None:
        return False, 0, 0

    #=========================================
    # 선분들의 기울기 값을 각각 모두 구한 후에 리스트에 담음. 
    # 기울기의 절대값이 너무 작은 경우 (수평선에 가까운 경우)
    # 해당 선분을 빼고 담음. 
    #=========================================
    slopes = []
    filtered_lines = []

    for line in all_lines:
        x1, y1, x2, y2 = line[0]

        if (x2 == x1):
            slope = 1000.0
        else:
            slope = float(y2-y1) / float(x2-x1)
    
        if 0.2 < abs(slope):
            slopes.append(slope)
            filtered_lines.append(line[0])

    if len(filtered_lines) == 0:
        return False, 0, 0

    #=========================================
    # 왼쪽 차선에 해당하는 선분과 오른쪽 차선에 해당하는 선분을 구분하여 
    # 각각 별도의 리스트에 담음.
    #=========================================
    left_lines = []
    right_lines = []

    for j in range(len(slopes)):
        Line = filtered_lines[j]
        slope = slopes[j]

        x1,y1, x2,y2 = Line

        # 기울기 값이 음수이고 화면의 왼쪽에 있으면 왼쪽 차선으로 분류함
        # 기준이 되는 X좌표값 = (화면중심값 - Margin값)
        Margin = 0
        
        if (slope < 0) and (x2 < WIDTH/2-Margin):
            left_lines.append(Line.tolist())

        # 기울기 값이 양수이고 화면의 오른쪽에 있으면 오른쪽 차선으로 분류함
        # 기준이 되는 X좌표값 = (화면중심값 + Margin값)
        elif (slope > 0) and (x1 > WIDTH/2+Margin):
            right_lines.append(Line.tolist())

    # 디버깅을 위해 차선과 관련된 직선과 선분을 그리기 위한 도화지 준비
    line_draw_img = roi_img.copy()
    
    # 왼쪽 차선에 해당하는 선분은 빨간색으로 표시
    for line in left_lines:
        x1,y1, x2,y2 = line
        cv2.line(line_draw_img, (x1,y1), (x2,y2), Red, 2)

    # 오른쪽 차선에 해당하는 선분은 노란색으로 표시
    for line in right_lines:
        x1,y1, x2,y2 = line
        cv2.line(line_draw_img, (x1,y1), (x2,y2), Yellow, 2)

    #=========================================
    # 왼쪽/오른쪽 차선에 해당하는 선분들의 데이터를 적절히 처리해서 
    # 왼쪽차선의 대표직선과 오른쪽차선의 대표직선을 각각 구함.
    # 기울기와 Y절편값으로 표현되는 아래와 같은 직선의 방적식을 사용함.
    # (직선의 방정식) y = mx + b (m은 기울기, b는 Y절편)
    #=========================================

    # 왼쪽 차선을 표시하는 대표직선을 구함        
    m_left, b_left = 0.0, 0.0
    x_sum, y_sum, m_sum = 0.0, 0.0, 0.0

    # 왼쪽 차선을 표시하는 선분들의 기울기와 양끝점들의 평균값을 찾아 대표직선을 구함
    size = len(left_lines)
    if size != 0:
        for line in left_lines:
            x1, y1, x2, y2 = line
            x_sum += x1 + x2
            y_sum += y1 + y2
            if(x2 != x1):
                m_sum += float(y2-y1)/float(x2-x1)
            else:
                m_sum += 0                
            
        x_avg = x_sum / (size*2)
        y_avg = y_sum / (size*2)
        m_left = m_sum / size
        b_left = y_avg - m_left * x_avg

        if m_left != 0.0:
            #=========================================
            # (직선 #1) y = mx + b 
            # (직선 #2) y = 0
            # 위 두 직선의 교점의 좌표값 (x1, 0)을 구함.           
            x1 = int((0.0 - b_left) / m_left)

            #=========================================
            # (직선 #1) y = mx + b 
            # (직선 #2) y = ROI_HEIGHT
            # 위 두 직선의 교점의 좌표값 (x2, ROI_HEIGHT)을 구함.               
            x2 = int((ROI_HEIGHT - b_left) / m_left)

            # 두 교점, (x1,0)과 (x2, ROI_HEIGHT)를 잇는 선을 그림
            cv2.line(line_draw_img, (x1,0), (x2,ROI_HEIGHT), Blue, 2)

    # 오른쪽 차선을 표시하는 대표직선을 구함      
    m_right, b_right = 0.0, 0.0
    x_sum, y_sum, m_sum = 0.0, 0.0, 0.0

    # 오른쪽 차선을 표시하는 선분들의 기울기와 양끝점들의 평균값을 찾아 대표직선을 구함
    size = len(right_lines)
    if size != 0:
        for line in right_lines:
            x1, y1, x2, y2 = line
            x_sum += x1 + x2
            y_sum += y1 + y2
            if(x2 != x1):
                m_sum += float(y2-y1)/float(x2-x1)
            else:
                m_sum += 0     
       
        x_avg = x_sum / (size*2)
        y_avg = y_sum / (size*2)
        m_right = m_sum / size
        b_right = y_avg - m_right * x_avg

        if m_right != 0.0:
            #=========================================
            # (직선 #1) y = mx + b 
            # (직선 #2) y = 0
            # 위 두 직선의 교점의 좌표값 (x1, 0)을 구함.           
            x1 = int((0.0 - b_right) / m_right)

            #=========================================
            # (직선 #1) y = mx + b 
            # (직선 #2) y = ROI_HEIGHT
            # 위 두 직선의 교점의 좌표값 (x2, ROI_HEIGHT)을 구함.               
            x2 = int((ROI_HEIGHT - b_right) / m_right)

            # 두 교점, (x1,0)과 (x2, ROI_HEIGHT)를 잇는 선을 그림
            cv2.line(line_draw_img, (x1,0), (x2,ROI_HEIGHT), Blue, 2)

    #=========================================
    # 차선의 위치를 찾기 위한 기준선(수평선)은 아래와 같음.
    #   (직선의 방정식) y = L_ROW 
    # 위에서 구한 2개의 대표직선, 
    #   (직선의 방정식) y = (m_left)x + (b_left)
    #   (직선의 방정식) y = (m_right)x + (b_right)
    # 기준선(수평선)과 대표직선과의 교점인 x_left와 x_right를 찾음.
    #=========================================

    #=========================================        
    # 대표직선의 기울기 값이 0.0이라는 것은 직선을 찾지 못했다는 의미임
    # 이 경우에는 교점 좌표값을 기존 저장해 놨던 값으로 세팅함 
    #=========================================
    if m_left == 0.0:
        x_left = prev_x_left  # 변수에 저장해 놓았던 이전 값을 가져옴

    #=========================================
    # 아래 2개 직선의 교점을 구함
    # (직선의 방정식) y = L_ROW  
    # (직선의 방정식) y = (m_left)x + (b_left)
    #=========================================
    else:
        x_left = int((L_ROW - b_left) / m_left)
                        
    #=========================================
    # 대표직선의 기울기 값이 0.0이라는 것은 직선을 찾지 못했다는 의미임
    # 이 경우에는 교점 좌표값을 기존 저장해 놨던 값으로 세팅함 
    #=========================================
    if m_right == 0.0:
        x_right = prev_x_right  # 변수에 저장해 놓았던 이전 값을 가져옴	
	
    #=========================================
    # 아래 2개 직선의 교점을 구함
    # (직선의 방정식) y = L_ROW  
    # (직선의 방정식) y = (m_right)x + (b_right)
    #=========================================
    else:
        x_right = int((L_ROW - b_right) / m_right)
       
    #=========================================
    # 대표직선의 기울기 값이 0.0이라는 것은 직선을 찾지 못했다는 의미임
    # 이 경우에 반대쪽 차선의 위치 정보를 이용해서 내 위치값을 정함 
    #=========================================
    if m_left == 0.0 and m_right != 0.0:
        x_left = x_right - 380

    if m_left != 0.0 and m_right == 0.0:
        x_right = x_left + 380

    # 이번에 구한 값으로 예전 값을 업데이트 함			
    prev_x_left = x_left
    prev_x_right = x_right
	
    # 왼쪽 차선의 위치와 오른쪽 차선의 위치의 중간 위치를 구함
    x_midpoint = (x_left + x_right) // 2 

    #=========================================
    # 디버깅용 이미지 그리기
    # (1) 수평선 그리기 (직선의 방정식) y = L_ROW 
    # (2) 수평선과 왼쪽 대표직선과의 교점 위치에 작은 녹색 사각형 그리기 
    # (3) 수평선과 오른쪽 대표직선과의 교점 위치에 작은 녹색 사각형 그리기 
    # (4) 왼쪽 교점과 오른쪽 교점의 중점 위치에 작은 파란색 사각형 그리기
    # (5) 화면의 중앙점 위치에 작은 빨간색 사각형 그리기 
    #=========================================
    cv2.line(line_draw_img, (0,L_ROW), (WIDTH,L_ROW), Yellow, 2)
    cv2.rectangle(line_draw_img, (x_left-5,L_ROW-5), (x_left+5,L_ROW+5), Green, 4)
    cv2.rectangle(line_draw_img, (x_right-5,L_ROW-5), (x_right+5,L_ROW+5), Green, 4)
    cv2.rectangle(line_draw_img, (x_midpoint-5,L_ROW-5), (x_midpoint+5,L_ROW+5), Blue, 4)
    cv2.rectangle(line_draw_img, (View_Center-5,L_ROW-5), (View_Center+5,L_ROW+5), Red, 4)

    # 위 이미지를 디버깅용 display_img에 overwrite해서 화면에 디스플레이 함
    display_img[ROI_START_ROW:ROI_END_ROW, 0:WIDTH] = line_draw_img
    cv2.imshow("Lanes positions", display_img)
    cv2.waitKey(1)

    return True, x_left, x_right

#=============================================
# AR 패지키지가 발행하는 토픽을 받아서 
# 제일 가까이 있는 AR Tag에 적힌 ID 값을 반환함.
# 추가로 거리값과 좌우치우침값을 함께 반환함.
#=============================================
def check_AR():
    global ar_msg

    if (len(ar_msg["ID"]) == 0):
        # 아직 AR 토픽이 없거나 발견된 AR태그가 없으면 리턴
        return 99, 10.0, 0.0, 0.0  # ID값은 99, 거리값은 10.0, Z위치값은 0.0, X위치값은 0.0으로 반환 

    # 새로 도착한 AR태그에 대해서 아래 작업 수행
    z_pos = 10.0  # Z위치값을 10미터로 초기화
    x_pos = 10.0  # X위치값을 10미터로 초기화
    
    for i in range(len(ar_msg["ID"])):
        # 발견된 AR태그 모두에 대해서 조사
        if(ar_msg["DZ"][i] < z_pos):
            # 더 가까운 거리에 AR태그가 있으면 그걸 사용
            id_value = ar_msg["ID"][i]
			
            # 카메라 왜곡때문에 거리값에 오차가 생길 수 있다. 여기서는 0.99 곱해서 보정함.
            z_pos = ar_msg["DZ"][i] * 0.99  ###############고쳐야함
            x_pos = ar_msg["DX"][i] * 0.99 ###########고쳐야함 자로 재바라= 줄여야 될듯 17cm을 19로 본다고 함

    # ID번호, 거리값(미터), 좌우치우침값(미터) 리턴
    distance = math.sqrt(z_pos**2 + x_pos**2)  # 거리값은 피타고라스 정리로 계산 
    return id_value, round(distance,2), round(z_pos,2), round(x_pos,2)

#=============================================
# 카메라로 AR태그 보면서 좌우 핸들링하며 주행
# AR태그를 왼쪽에 끼고 주행한다
#=============================================
def AR_drive():

    # check_AR() 이용해서 AR태그의 ID값, 거리값, Z값, X값을 구한다.    
    ar_ID, distance, z_pos, x_pos = check_AR()
    print("Distance=",distance,"Z_pos=",z_pos," X_pos=",x_pos)
    
    # ID값이 99라는 건 AR태그를 발견하지 못했다는 의미 	
    if (ar_ID == 99):
        found = False
        drive_angle = new_angle    
        return found, drive_angle  # drive_angle에 기존 값을 넣어서 리턴 
    
    # AR태그까지의 거리가 1미터보다 멀면 xx_pos와 drive_angle에 각각 적당한 값 넣기 
    if (distance > 1.0):   
        xx_pos = x_pos + 0.2  
        drive_angle = int(xx_pos*40) 

    # AR태그까지의 거리가 70센치~1미터 사이면 xx_pos와 drive_angle에 각각 적당한 값 넣기 
    elif (distance > 0.7):
        xx_pos = x_pos + 0.3  
        drive_angle = int(xx_pos*60)  ### 자동차와 테크와 가까우면 거를 두어야 함

    # AR태그까지의 거리가 30센치~70센치 사이면 xx_pos와 drive_angle에 각각 적당한 값 넣기 
    elif (distance > 0.3):
        xx_pos = x_pos + 0.5 
        drive_angle = int(xx_pos*80) 

    # 위의 경우가 아니면 (AR태그까지의 거리가 30센치보다 작으면) xx_pos와 drive_angle에 각각 적당한 값 넣기 
    else:
        xx_pos = x_pos + 0.7 
        drive_angle = int(xx_pos*100) 

    found = True
    return found, drive_angle  # AR태그를 발견했는지 여부(True, False)와 핸들조향각 값을 리턴

# ====================================
# 카메라에 잡힌 영상에서 King 카드와 Ace 카드를 찾는다.
# 두 카드의 위치관계에 따라 -1,0,+1 값을 반환한다.
# ====================================
def object_detect():

    global width, height
    input_mean = 127.5
    input_std = 127.5
    min_conf_threshold = 0.5
    imW, imH = 640, 480

    # TensorFlow 라이브러리를 임포트한다
    pkg = importlib.util.find_spec('tflite_runtime')
    if pkg:
        # 차량에 tflite_runtime가 설치되어 있으므로 여기가 실행된다
        from tflite_runtime.interpreter import Interpreter
    else:
        from tensorflow.lite.python.interpreter import Interpreter

    # labelmap.txt 파일을 읽어들인다
    with open(PATH_TO_LABELS, 'r') as f:
        labels = [line.strip() for line in f.readlines()]

    # 학습된 Weight파일(학습모델)을 가져다가 interpreter를 생성한다 
    interpreter = Interpreter(model_path=PATH_TO_CKPT)
    interpreter.allocate_tensors()

    # 모델의 이런저런 설정값들을 읽어들인다
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    height = input_details[0]['shape'][1]
    width = input_details[0]['shape'][2]
    floating_model = (input_details[0]['dtype'] == np.float32)
    input_mean = 127.5
    input_std = 127.5

    # output layer name을 체크해서 모델이 TF2, TF1 어느 걸로 만들어진 것인지 파악한다
    outname = output_details[0]['name']

    if ('StatefulPartitionedCall' in outname):
        # TF2 모델인 경우 (이번엔 이게 맞다)
        boxes_idx, classes_idx, scores_idx = 1, 3, 0
    else: 
        # TF1 모델인 경우
        boxes_idx, classes_idx, scores_idx = 0, 1, 2

    # 초당 몇장의 사진을 처리할 수 있는지 표시하기 위해 FPS 계산을 준비한다
    frame_rate_calc = 1
    freq = cv2.getTickFrequency()
   
    ace_found, king_found, ace_king_found = False, False, False
    ace_position, king_position = 0, 0
    
    # Ace 또는 King 그림을 찾을 때까지 계속 인식을 시도한다.
    while (ace_king_found == False):
    
        # FPS 계산을 위해 타이머를 시작
        t1 = cv2.getTickCount()

        # 카메라 이미지를 가져다가 크기를 변경한다 [1xHxWx3]
        frame = image.copy()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (width, height))
        input_data = np.expand_dims(frame_resized, axis=0)

        # floating model을 사용하고 있다면 (그렇다) 픽셀 값을 Normalize 한다 
        if floating_model:  # 이거 True 이다.
            input_data = (np.float32(input_data) - input_mean) / input_std

        # 이미지에서 객체 찾는 작업을 진행한다
        interpreter.set_tensor(input_details[0]['index'],input_data)
        interpreter.invoke()

        # ====================================
        # 객체 인식 결과를 가져온다
        # boxes에는 인식된 객체를 둘러싼 사각형의 좌표값이 담긴다.
        # classes에는 인식된 객체의 종류정보(사람, 자전거 등의 클래스정보)가 담긴다.
        # scores에는 인식결과에 대한 인식확률값(100% 확신? 75% 확신?)이 담긴다. 
        # ====================================
        boxes = interpreter.get_tensor(output_details[boxes_idx]['index'])[0] 
        classes = interpreter.get_tensor(output_details[classes_idx]['index'])[0] 
        scores = interpreter.get_tensor(output_details[scores_idx]['index'])[0] 

        # 인식확률이 정해진 값보다 크면 제대로 인식한 것으로 치고 루프를 돌면서 인식된 객체의 정보를 수집한다
        detected_card_num = 0
        for i in range(len(scores)):
            if ((scores[i] > min_conf_threshold) and (scores[i] <= 1.0)):

                # 바운딩박스의 좌표값을 얻어서 사각형 그릴 좌표값을 계산한다
                # 화면 크기를 벗어날 수 있으므로 가로세로 최대 크기 안으로 좌표값을 제한한다
                ymin = int(max(1,(boxes[i][0] * imH)))
                xmin = int(max(1,(boxes[i][1] * imW)))
                ymax = int(min(imH,(boxes[i][2] * imH)))
                xmax = int(min(imW,(boxes[i][3] * imW)))
                
                # 사각형 테두리를 연두색으로 그린다
                cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), (10,255,0), 2)

                # 라벨을 적는다  'person: 72%'와 같은 형태로 적는다
                object_name = labels[int(classes[i])] 
                label = '%s: %d%%' % (object_name, int(scores[i]*100)) 
                labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2) 
                label_ymin = max(ymin, labelSize[1] + 10) 
                cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) 
                cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) 

                # Ace 또는 King 카드를 발견하면 바운딩박스의 중앙지점의 X좌표값을 계산한다
                if object_name == 'ace':
                    ace_found = True
                    ace_position = (xmin+xmax)//2
                    print("ACE found:", ace_position)
                                           
                if object_name == 'king':
                    king_found = True
                    king_position = (xmin+xmax)//2
                    print("KING found:", king_position)
     
                detected_card_num = detected_card_num + 1

        # 총 몇 장의 카드를 찾았는지 출력한다. 
        print("#################", detected_card_num, "cards")
		
        # 발견한 카드가 총 2장이고 하나가 Ace 카드, 다른 하나가 King 카드이면 ace_king_found를 True로 한다.
        if (detected_card_num==2) and (ace_found == True) and (king_found == True):
            ace_king_found = True
        
        # 화면 구석에 FPS 정보 출력한다
        cv2.putText(frame,'FPS: {0:.2f}'.format(frame_rate_calc),(30,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2,cv2.LINE_AA)

        # 화면에 객체인식 결과 표시가 추가된 그림을 표시한다
        cv2.imshow('Object detector', frame)
        cv2.waitKey(1)
        
        # FPS 계산한다
        t2 = cv2.getTickCount()
        time1 = (t2-t1)/freq
        frame_rate_calc= 1/time1
 
    if (king_position < ace_position):
        # 오른쪽에 ACE 카드가 있는 경우 1값을 리턴한다 
        return 1

    elif (ace_position < king_position):
        # 화면 왼쪽에 ACE 카드가 있는 경우 -1값을 리턴한다
        return -1

    else:
        # 이도저도 아니면 0값을 리턴한다 
        return 0

#=============================================
# 차량을 이동시키는 함수.  
# 입력으로 조향각을 받아 2초간 차량을 이동시킴.
#=============================================
def move_car(move_angle, speed, X):
    for i in range(X): 
        drive(angle=move_angle, speed = speed ) 
        time.sleep(0.01)
        
#=============================================
# 실질적인 메인 함수 
#=============================================
def start():

    global motor, ultra_msg, image, img_ready 
    global new_angle, new_speed
    global interpreter, PATH_TO_CKPT, PATH_TO_LABELS 
    
    SENSOR_DRIVE = 1
    TRAFFIC_SIGN = 2
    LANE_DRIVE = 3
    AR_DRIVE = 4
    OBJECT_DETECT = 5
    PARKING = 6
    FINISH = 9
	
    # 처음에 어떤 미션부터 수행할 것인지 여기서 결정한다. 
    drive_mode = TRAFFIC_SIGN   # SENSOR_DRIVE
    
    #=========================================
    # 노드를 생성하고, 구독/발행할 토픽들을 선언합니다.
    #=========================================
    rospy.init_node('Track_Driver')
    motor = rospy.Publisher('xycar_motor', xycar_motor, queue_size=1)
    rospy.Subscriber("xycar_ultrasonic", Int32MultiArray, ultra_callback, queue_size=1)
    rospy.Subscriber("/usb_cam/image_raw/",Image,usbcam_callback, queue_size=1)
    rospy.Subscriber('ar_pose_marker', AlvarMarkers, ar_callback, queue_size=1 )

    # 객체인식에 파일들이 어디에 있는지 경로 정보를 launch 파일에서 읽어들인다. 
    PATH_TO_CKPT = rospy.get_param('~ckpt_path')
    PATH_TO_LABELS = rospy.get_param('~label_path')

    #=========================================
    # 첫번째 토픽이 도착할 때까지 기다립니다.
    #=========================================
    rospy.wait_for_message("/usb_cam/image_raw/", Image)
    print("Camera Ready --------------")
    rospy.wait_for_message("xycar_ultrasonic", Int32MultiArray)
    print("UltraSonic Ready ----------")
    rospy.wait_for_message("ar_pose_marker", AlvarMarkers)
    print("AR detector Ready ----------")

    print("======================================")
    print(" S T A R T    D R I V I N G ...")
    print("======================================")
	
    # 일단 차량이 움직이지 않도록 정지 상태로 만듭니다.  
    stop_car(1)
	
    #=========================================
    # 메인 루프 
    #=========================================
    while not rospy.is_shutdown():

        # ======================================
        # 출발선에서 신호등을 찾습니다. 
        # 일단 정차해 있다가 파란색 불이 켜지면 출발합니다.
        # 초음파센서 미로주행 SENSOR_DRIVE 모드로 넘어갑니다.  
        # ======================================
        while drive_mode == TRAFFIC_SIGN:
		
            # 앞에 있는 신호등에 파란색 불이 켜졌는지 체크합니다.  
            result = check_traffic_sign()
			
            if (result == True):
                # 만약 신호등이 파란색 불로 바뀌었으면 미로주행 SENSOR_DRIVE 모드로 넘어갑니다.
                drive_mode = SENSOR_DRIVE  
                print ("----- Ultrasonic driving Start... -----")
                cv2.destroyAllWindows()
                
        # ======================================
        # 초음파 센서로 주행합니다.
        # 정지선이 보이면 차선인식주행 LANE_DRIVE 모드로 넘어갑니다. 
        # ======================================
        while drive_mode == SENSOR_DRIVE:

            # 초음파세센서를 이용해서 미로주행을 합니다. 
            sonic_drive()  

            # 바닥에 정지선이 있는지 체크합니다. 
            result = check_stopline()
             
            if (result == True):
                # 만약 정지선이 발견되면 차량을 정차시키고 LANE_DRIVE 모드로 넘거갑니다.
                stop_car(1)
                drive_mode = LANE_DRIVE  
                print ("----- Lane driving Start... -----")
                cv2.destroyAllWindows()
                
        # ======================================
        # 차선을 보고 주행합니다.
        # AR태그가 발견되면 다른 주행모드로 넘어갑니다. 
		# 발견된 AR태그의 ID가 '1'이면 AR주행 모드로 (AR_DRIVE) 넘어갑니다. 
		# 발견된 AR태그의 ID가 '2'이면 객체인식 모드로 (OBJECT_DETECT) 넘어갑니다. 		 
        # ======================================
        while drive_mode == LANE_DRIVE:
		
            # 카메라 이미지에서 차선의 위치를 알아냅니다. 
            found, x_left, x_right = lane_detect()
			
            if found:
                # 차선인식이 됐으면 차선의 위치정보를 이용해서 핸들 조향각을 결정합니다. 
                x_midpoint = (x_left + x_right) // 2 
                new_angle = (x_midpoint - View_Center) // 2
                if abs(new_angle) < 10:
                    new_speed =  25
                elif abs(new_angle) > 30 :  ##  이거는 나중에 해보자
                    new_speed = 17


                drive(new_angle, new_speed)  
				
            else:
                # 차선인식이 안 됐으면 기존 핸들 각도를 그대로 유지한채로 주행합니다. 	
                drive(new_angle, new_speed)
            
            # 전방에 AR태그가 보이는지 체크합니다.             
            ar_ID, distance, z_pos, x_pos = check_AR()

            if (distance < 0.75) and (ar_ID == 1): # AR_Drive Sign
                # 50센치 안에 AR태그가 있고 ID값이 '1'이면 AR주행 AR_DRIVE 모드로 넘어갑니다.  
                cv2.destroyAllWindows()
                print("AR_ID=",ar_ID," Distance=",distance," Z_pos=",z_pos," X_pos=",x_pos)
                drive_mode = AR_DRIVE  
                new_angle = -40
                # stop_car(1)
                print ("----- AR driving Start... -----")

            elif (distance < 0.95) and (ar_ID == 2): # Obejct Detection Sign
                # 50센치 안에 AR태그가 있고 ID값이 '2'이면 객체인식 OBJECT_DETECT 모드로 넘어갑니다.  
                stop_car(1)
                drive_mode = OBJECT_DETECT  
                print ("----- Object Detection Start... -----")
                
        # ======================================
        # AR 표지판을 따라서 주행합니다. 
        # AR 표지판이 더 이상 없으면 다시 차선주행 LANE_DRIVE 모드로 변경합니다.
        # ======================================
        retry_count = 0
        while drive_mode == AR_DRIVE:
		
            # AR태그를 찾고 핸들각도를 결정합니다. 
            found, new_angle = AR_drive()

            if found:
                # AR태그가 인식되었으면 적절하게 핸들을 꺾어 주행합니다. 
                print("AR drive", new_angle)
                drive(new_angle, 24) #### fifiten
                retry_count = 0
				
            else:
                # AR태그가 보이지 않는다면 5번의 재시도를 해봅니다. 
                retry_count = retry_count + 1 
				
                if(retry_count < 8):
                    # 5번의 재시도 중에는 기존 핸들각도를 그대로 유지합니다. 
                    print("Keep going...", new_angle)
                    drive(new_angle, new_speed)
					
                else:
                    # 5번의 재시도가 모두 끝나면 AR태그 찾기를 그만두고 LANE_DRIVE 모드로 넘어갑니다. 
                    drive_mode = LANE_DRIVE  
                    print ("----- Lane driving Start... -----")

        # ======================================
        # 앞에 놓여져 있는 그림을 보고 ACE와 King 카드를 찾습니다.
        # 차량을 ACE 카드가 있는 방향으로 조금 주행시키고 차를 세웁니다.  
        # ======================================
        while drive_mode == OBJECT_DETECT:
		
            # 객체인식 작업을 진행하고 그 결과값을 얻어옵니다.            
            object_position = object_detect()
            
            if (object_position == -1):
                # ACE 카드가 왼쪽에 있는 것으로 판단되면 왼쪽으로 좌회전해서 차를 세웁니다.
                print("....Ace card is on Left")
                move_car(move_angle = -100, speed = 24, X = 150)
                move_car(move_angle =  50,  speed = 22, X = 100)
                # move_car(move_angle =  100,  speed = 15, X = 100)
                drive_mode = PARKING
				
            elif (object_position == +1):
                # ACE 카드가 오른쪽에 있는 것으로 판단되면 오른쪽으로 우회전해서 차를 세웁니다.
                print("....Ace card is on Right")
                move_car(move_angle = 100,  speed = 24, X = 150)
                move_car(move_angle =  -50,  speed = 22, X = 100)
                # move_car(move_angle =  -100,  speed = 15, X = 100)
                drive_mode = PARKING
				
            else:
                # 이도저도 아니면 객체인식을 다시 시도합니다. 
                print("....Card Detecting...")
                
        # ======================================
        # AR 표지판을 보고 주차합니다. 
        # AR 표지판 바로 앞에 가면 주행종료 모드로 변경합니다.
        # ======================================
        while drive_mode == PARKING:
       
            # 전방에 AR태그가 보이는지 체크합니다.   
            ar_ID, distance, z_pos, x_pos = check_AR()
            print("Distance=",distance,"Z_pos=",z_pos," X_pos=",x_pos)

            if (ar_ID == 99):
                # AR태그가 안 보이면 AR태그를 계속 찾습니다.   
                continue
    
            else:
                # AR태그가 안 보이면 AR태그를 계쏙 찾습니다.   
                if (z_pos > 0.6):
                    # Z값이 20센치 이상이이면 핸들 각도를 조종하면서 주행합니다.  
                    new_angle = int(x_pos // 0.004)
					
                else:
                    # Z값이 20센치 이하이면 주차영역에 들어온 것으로 하고 차량을 세우고 종료 FINISH 모드로 넘어갑니다.     
                    new_angle = 0
                    new_speed = 0                      
                    drive_mode = FINISH  
                    print ("----- Parking completed... -----")

            drive(new_angle, new_speed)

        # ======================================
        # 주행을 끝냅니다. 
        # ======================================
        if drive_mode == FINISH:
           
            # 차량을 정지시키고 모든 작업을 끝냅니다.
            stop_car(1)  
            time.sleep(2)
            print ("----- Bye~! -----")
            return            

#=============================================
# 메인함수 호툴
# start() 함수가 실질적인 메인함수임. 
#=============================================
if __name__ == '__main__':
    start()
    

# 82.2 느린 천천히
# 너를 믿자 손건희 화이팅 
# 아침 6시 9분 최종 수정본









