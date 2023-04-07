import math
import argparse
import pygame
import os
import sys
import time
import sqlite3
import argparse
import numpy as np
from rockx import RockX
import cv2
import socket


class FaceDB:

    def __init__(self, db_file):
        self.db_file = db_file
        self.conn = sqlite3.connect(self.db_file)
        self.cursor = self.conn.cursor()
        if not self._is_face_table_exist():
            self.cursor.execute("create table FACE (NAME text, VERSION int, FEATURE blob, ALIGN_IMAGE blob)")

    def load_face(self):
        all_face = dict()
        c = self.cursor.execute("select * from FACE")
        for row in c:
            name = row[0]
            version = row[1]
            feature = np.frombuffer(row[2], dtype='float32')
            align_img = np.frombuffer(row[3], dtype='uint8')
            align_img = align_img.reshape((112, 112, 3))
            all_face[name] = {
                'feature': RockX.FaceFeature(version=version, len=feature.size, feature=feature),
                'image': align_img
            }
        return all_face

    def insert_face(self, name, feature, align_img):
        self.cursor.execute("INSERT INTO FACE (NAME, VERSION, FEATURE, ALIGN_IMAGE) VALUES (?, ?, ?, ?)",
                            (name, feature.version, feature.feature.tobytes(), align_img.tobytes()))
        self.conn.commit()

    def _get_tables(self):
        cursor = self.cursor
        cursor.execute("select name from sqlite_master where type='table' order by name")
        tables = cursor.fetchall()
        return tables

    def _is_face_table_exist(self):
        tables = self._get_tables()
        for table in tables:
            if 'FACE' in table:
                return True
        return False


def get_max_face(results):
    max_area = 0
    max_face = None
    for result in results:
        area = (result.box.bottom - result.box.top) * (result.box.right * result.box.left)
        if area > max_area:
            max_face = result
    return max_face


def get_face_feature(image_path):
    img = cv2.imread(image_path)
    img_h, img_w = img.shape[:2]
    ret, results = face_det_handle.rockx_face_detect(img, img_w, img_h, RockX.ROCKX_PIXEL_FORMAT_BGR888)
    if ret != RockX.ROCKX_RET_SUCCESS:
        return None, None
    max_face = get_max_face(results)
    if max_face is None:
        return None, None
    ret, align_img = face_landmark5_handle.rockx_face_align(img, img_w, img_h,
                                                            RockX.ROCKX_PIXEL_FORMAT_BGR888,
                                                            max_face.box, None)
    if ret != RockX.ROCKX_RET_SUCCESS:
        return None, None
    if align_img is not None:
        ret, face_feature = face_recog_handle.rockx_face_recognize(align_img)
        if ret == RockX.ROCKX_RET_SUCCESS:
            return face_feature, align_img
    return None, None


def get_all_image(image_path):
    img_files = dict()
    g = os.walk(image_path)

    for path, dir_list, file_list in g:
        for file_name in file_list:
            file_path = os.path.join(path, file_name)
            if not os.path.isdir(file_path):
                img_files[os.path.splitext(file_name)[0]] = file_path
    return img_files


def import_face(face_db, images_dir):
    image_files = get_all_image(images_dir)
    image_name_list = list(image_files.keys())
    for name, image_path in image_files.items():
        feature, align_img = get_face_feature(image_path)
        if feature is not None:
            face_db.insert_face(name, feature, align_img)
            print('[%d/%d] success import %s ' % (image_name_list.index(name)+1, len(image_name_list), image_path))
        else:
            print('[%d/%d] fail import %s' % (image_name_list.index(name)+1, len(image_name_list), image_path))


def search_face(face_library, cur_feature):
    min_similarity = 10.0
    target_name = None
    target_face = None
    for name, face in face_library.items():
        feature = face['feature']
        ret, similarity = face_recog_handle.rockx_face_similarity(cur_feature, feature)
        if similarity < min_similarity:
            target_name = name
            min_similarity = similarity
            target_face = face
    if min_similarity < 1.0:
        return target_name, min_similarity, target_face
    return None, -1, None


if __name__ == '__main__':
    sser=socket.socket()
    sser.bind(("0.0.0.0" ,5555))
    sser.listen(3)
    conn, addr = sser.accept()

    parser = argparse.ArgumentParser(description="RockX Pose Demo")
    parser = argparse.ArgumentParser(description="RockX Face Recognition Demo")
    parser.add_argument('-c', '--camera', help="camera index", type=int, default=10)
    parser.add_argument('-b', '--db_file', help="face database path", required=True)
    parser.add_argument('-d', '--device', help="target device id", type=str)
    args = parser.parse_args()

    pose_body_handle = RockX(RockX.ROCKX_MODULE_POSE_BODY, target_device=args.device)
    face_det_handle = RockX(RockX.ROCKX_MODULE_FACE_DETECTION, target_device=args.device)
    face_landmark5_handle = RockX(RockX.ROCKX_MODULE_FACE_LANDMARK_5, target_device=args.device)
    face_recog_handle = RockX(RockX.ROCKX_MODULE_FACE_RECOGNIZE, target_device=args.device)
    face_track_handle = RockX(RockX.ROCKX_MODULE_OBJECT_TRACK, target_device=args.device)

    face_db = FaceDB(args.db_file)

    # load face from database
    face_library = face_db.load_face()
    print("load %d face" % len(face_library))

    cap = cv2.VideoCapture(args.camera)
    cap.set(3, 1280)
    cap.set(4, 720)
    last_face_feature = None
    recogyes=0#人脸识别成功变量
    timer_on=0 #计时器开启变量
    se=0 
    start=0
    startrecog=0#开始识别人脸
    choose=0 #选择该运动
    success_time=0
    #警告初始化
    warning_time=0 
    completion_degree=0 #完成度初始化
    justdone=0#刚刚做完
    #新月式
    moon=0
    #女神式
    Goddess=0
    #风吹树式
    tree=0
    pose=0
    #engine = pyttsx3.init()#初始化
    pygame.mixer.init()#初始化
    pygame.mixer.music.load('1.wav')  
    pygame.mixer.music.set_volume(1.5) 

    while True:
        ret, frame = cap.read()
        in_img_h, in_img_w = frame.shape[:2]
        

        if recogyes==0 :

            ret, results = face_det_handle.rockx_face_detect(frame, in_img_w, in_img_h, RockX.ROCKX_PIXEL_FORMAT_BGR888)

            ret, results = face_track_handle.rockx_object_track(in_img_w, in_img_h, 3, results)

        
            for result in results:
                # face align
                ret, align_img = face_landmark5_handle.rockx_face_align(frame, in_img_w, in_img_h,
                                                                     RockX.ROCKX_PIXEL_FORMAT_BGR888,
                                                                     result.box, None)

                # get face feature
                if ret == RockX.ROCKX_RET_SUCCESS and align_img is not None:
                    ret, face_feature = face_recog_handle.rockx_face_recognize(align_img)

                # search face
                if ret == RockX.ROCKX_RET_SUCCESS and face_feature is not None:
                    target_name, diff, target_face = search_face(face_library, face_feature)
                    recogyes=1
                    print('recognize success')
                    conn.send("0".encode())
                    # sser.close()
                    if target_name == 'lml' :
                        print('hello lml')
        
        if justdone==1 :
            justdone=justdone+1
            if justdone==15:
                justdone=0


        #人脸识别成功：
        if recogyes==1 and justdone==0 :
            conn, addr = sser.accept()
            
            while True:
            
                ret, frame = cap.read()
                in_img_h, in_img_w = frame.shape[:2]
            
                
                ret, results = pose_body_handle.rockx_pose_body(frame, in_img_w, in_img_h, RockX.ROCKX_PIXEL_FORMAT_BGR888)        

                index=0
                for result in results:
                    for p in result.points:
                        cv2.circle(frame, (p.x, p.y), 3, (0, 255, 0), 3)
                    for pairs in RockX.ROCKX_POSE_BODY_KEYPOINTS_PAIRS:
                        pt1 = result.points[pairs[0]] #一对中左边的点
                        pt2 = result.points[pairs[1]] #一对中右边的点
                        if pt1.x <= 0 or pt1.y <= 0 or pt2.x <= 0 or pt2.y <= 0:
                            continue
                        cv2.line(frame, (pt1.x, pt1.y), (pt2.x, pt2.y), (255, 0, 0), 2)

                    #计算躯干斜率
                    nose = result.points[0]
                    neck = result.points[1]
                    rsho = result.points[2]
                    relb = result.points[3]
                    rwr = result.points[4]
                    lsho = result.points[5]
                    lelb = result.points[6]
                    lwr = result.points[7]
                    rhip = result.points[8]
                    rknee = result.points[9]
                    rank = result.points[10]
                    lhip = result.points[11]
                    lknee = result.points[12]
                    lank = result.points[13]
                    reye = result.points[14]
                    leye = result.points[15]
                    rear = result.points[16]
                    lear = result.points[17]
                    #斜率初始化
                    m1=2000
                    m2=2000
                    m3=2000
                    m4=2000
                    m5=2000
                    m6=2000
                    m7=2000
                    m8=2000
                    m9=2000
                    m10=2000

                    pose=-1  

                    if (neck.x-rhip.x) != 0 :
                        m1=(neck.y-rhip.y) / (neck.x-rhip.x)
                    
                    if (rhip.x-rknee.x) != 0 :
                        m2=(rhip.y-rknee.y) / (rhip.x-rknee.x)
                    
                    if (neck.x-lhip.x) != 0:
                        m4=(neck.y-lhip.y) / (neck.x-lhip.x)
                    
                    if (lhip.x-lknee.x) != 0:
                        m5=(lhip.y-lknee.y) / (lhip.x-lknee.x)
                    
                    if (rknee.x-rank.x) != 0:
                        m3=(rknee.y-rank.y) / (rknee.x-rank.x)
                    
                    if (lknee.x-lank.x) != 0:
                        m6=(lknee.y-lank.y) / (lknee.x-lank.x)
                    
                    if (rsho.x-relb.x) != 0:
                        m7=(rsho.y-relb.y) / (rsho.x-relb.x)
                    
                    if (relb.x-rwr.x) != 0:
                        m8=(relb.y-rwr.y) / (relb.x-rwr.x)
                    
                    if (lsho.x-lelb.x) != 0:
                        m9=(lsho.y-lelb.y) / (lsho.x-lelb.x)
                    
                    if (lelb.x-lwr.x) != 0:
                        m10=(lelb.y-lwr.y) / (lelb.x-lwr.x)
                    

                    #判定女神式
                    if m2>=-3 and m2<=-0.1 and m5>=0.1 and m5<=3 and relb.y>rsho.y and lelb.y>lsho.y and rhip.x-rknee.x>=100:
                        pose=1
                    #门闩式
                    elif m5>0.1 and m5<9 and m6>0.1 and m6<9 and lknee.x-lhip.x>=90 and m7>-99 and m7<0 and m8>-99 and m8<0 and  rwr.y<relb.y and relb.y<rsho.y:                
                        pose=2
                    #风吹树式(右摆)
                    elif m7>=0.1 and m7<=10 and m8>=0.1 and m8<=10 and m9>=0.1 and m9<=10 and m10>=0.1 and m10<=10 and lwr.y<lelb.y and lelb.y<lsho.y:
                        pose=3
                    else:
                        pose=-1


                    #进入门闩式准备
                    if pose==2 and se==0:
                        se=1
                        moon=1 #判定进入新月式
                        #engine.say("door please hold on")
                        #engine.runAndWait()
                        print('Door_please hold on!')
                        conn.send("2".encode())
                        time.sleep(4)
                        timer_on=1

                    #进入女神式准备
                    if pose==1 and se==0 :
                        se=1
                        Goddess=1 #判定进入女神式
                        #engine.say("yoga goddess please hold on")
                        #engine.runAndWait()
                        print('Yoga_Goddess_please hold on!')
                        conn.send("1".encode())
                        time.sleep(4)
                        timer_on=1

                    #进入风吹树式准备
                    if pose==3 and se==0:
                        se=1
                        tree=1 #判定进入风吹树式
                        #engine.say("tree please hold on")
                        #engine.runAndWait()
                        print('Tree_please hold on!')
                        conn.send("3".encode())
                        time.sleep(4)
                        timer_on=1

                    #判定是否开启计时器
                    if timer_on==1:
                        begin_time=time.time()
                        timer_on=0
                        start=1
                        #engine.say("start time")
                        #engine.runAndWait()
                        print('start timing!')
                        pygame.mixer.music.play()
            

                    #新月式计时-15s
                    if start==1 and moon==1 and pose==2 :
                        success_time=success_time+1
                        end_time=time.time()
                        during=end_time-begin_time
                        print(during)
                        if during>=15:
                            completion_degree=(success_time)/(success_time+warning_time)
                            pygame.mixer.music.stop()
                            print('Door-Done! Congrulations!')
                            print(completion_degree)
                            if completion_degree<0.3:
                                conn.send("4".encode())
                            elif completion_degree<0.6:
                                conn.send("5".encode())
                            else:
                                conn.send("6".encode())
                            success_time=0
                            warning_time=0
                            start=0
                            se=0
                            moon=0
                            recogyes=0
                            justdone=1
                            break
                    elif start==1 and moon==1 and pose !=2:
                        warning_time=warning_time+1
                        print('Not standard')
                        end_time_2=time.time()
                        during=end_time_2-begin_time
                        if during>15:
                            completion_degree=(success_time)/(success_time+warning_time)
                            pygame.mixer.music.stop()
                            print('Door-Done! Congrulations!')
                            print(completion_degree)
                            if completion_degree<0.3:
                                conn.send("4".encode())
                            elif completion_degree<0.6:
                                conn.send("5".encode())
                            else:
                                conn.send("6".encode())
                            success_time=0
                            warning_time=0
                            start=0
                            se=0
                            moon=0
                            recogyes=0
                            justdone=1
                            break
                    #女神式计时-15s
                    if start==1 and Goddess==1 and pose==1 :
                        success_time=success_time+1
                        end_time=time.time()
                        during=end_time-begin_time
                        print(during)
                        if during>=15:
                            completion_degree=(success_time)/(success_time+warning_time)
                            pygame.mixer.music.stop()
                            print('Yoga_Goddess-Done! Congrulations!')
                            print(completion_degree)
                            if completion_degree<0.3:
                                conn.send("4".encode())
                            elif completion_degree<0.6:
                                conn.send("5".encode())
                            else:
                                conn.send("6".encode())
                            success_time=0
                            warning_time=0
                            start=0
                            se=0
                            Goddess=0
                            recogyes=0
                            justdone=1
                            break
                    elif start==1 and Goddess==1 and pose !=1:
                        warning_time=warning_time+1
                        print('Not standard')
                        end_time_2=time.time()
                        during=end_time_2-begin_time
                        if during>15:
                            completion_degree=(success_time)/(success_time+warning_time)
                            pygame.mixer.music.stop()
                            print('Yoga_Goddess-Done! Congrulations!')
                            print(completion_degree)
                            if completion_degree<0.3:
                                conn.send("4".encode())
                            elif completion_degree<0.6:
                                conn.send("5".encode())
                            else:
                                conn.send("6".encode())
                            success_time=0
                            warning_time=0
                            start=0
                            se=0
                            Goddess=0
                            recogyes=0
                            justdone=1
                            break
                    #风吹树式计时-15s
                    if start==1 and tree==1 and pose==3 :
                        success_time=success_time+1
                        end_time=time.time()
                        during=end_time-begin_time
                        print(during)
                        if during>=15:
                            completion_degree=(success_time)/(success_time+warning_time)
                            pygame.mixer.music.stop()
                            print('Tree-Done! Congrulations!')
                            print(completion_degree)
                            if completion_degree<0.3:
                                conn.send("4".encode())
                            elif completion_degree<0.6:
                                conn.send("5".encode())
                            else:
                                conn.send("6".encode())
                            success_time=0
                            warning_time=0
                            start=0
                            se=0
                            tree=0
                            recogyes=0
                            justdone=1
                            break
                    elif start==1 and tree==1 and pose !=3:
                        warning_time=warning_time+1
                        print('Not standard')
                        end_time_2=time.time()
                        during=end_time_2-begin_time
                        if during>15:
                            completion_degree=(success_time)/(success_time+warning_time)
                            pygame.mixer.music.stop()
                            print('Tree-Done! Congrulations!')
                            print(completion_degree)
                            if completion_degree<0.3:
                                conn.send("4".encode())
                            elif completion_degree<0.6:
                                conn.send("5".encode())
                            else:
                                conn.send("6".encode())
                            success_time=0
                            warning_time=0
                            start=0
                            se=0
                            tree=0
                            recogyes=0
                            justdone=1
                            break

                    
                    index += 1

                cv2.imshow('RockX Pose - ' + str(args.device), frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break


        if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

    pose_body_handle.release()
