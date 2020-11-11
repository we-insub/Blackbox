import datetime
import errno
import os
import threading
import time
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from calibration import calib, undistort
from threshold import gradient_combine, hls_combine, comb_result
from finding_lines import Line, warp_image, find_LR_lines, draw_lane, print_road_status, print_road_map
from skimage import exposure
input_type = 'video' #'video' # 'image' 인풋 타입 지정
input_name = 'challenge_video.mp4' #'test_images/straight_lines1.jpg' # 'challenge_video.mp4' 불러올 실제 파일의 이름과 확장자명

left_line = Line()
right_line = Line()

th_sobelx, th_sobely, th_mag, th_dir = (35, 100), (30, 255), (30, 255), (0.7, 1.3)
th_h, th_l, th_s = (10, 100), (0, 60), (85, 255)

# camera matrix & distortion coefficient
mtx, dist = calib()

frameCnt = 0  # 프레임을 저장할 변수
saveCnt = 0 # 세이브횟수를 저장할 변수

if __name__ == '__main__':

    if input_type == 'image':
        img = cv2.imread(input_name)
        undist_img = undistort(img, mtx, dist)
        undist_img = cv2.resize(undist_img, None, fx=1 / 2, fy=1 / 2, interpolation=cv2.INTER_AREA)
        rows, cols = undist_img.shape[:2]

        combined_gradient = gradient_combine(undist_img, th_sobelx, th_sobely, th_mag, th_dir)
        combined_hls = hls_combine(undist_img, th_h, th_l, th_s)
        combined_result = comb_result(combined_gradient, combined_hls)

        c_rows, c_cols = combined_result.shape[:2]
        s_LTop2, s_RTop2 = [c_cols / 2 - 24, 5], [c_cols / 2 + 24, 5]
        s_LBot2, s_RBot2 = [110, c_rows], [c_cols - 110, c_rows]

        src = np.float32([s_LBot2, s_LTop2, s_RTop2, s_RBot2])
        dst = np.float32([(170, 720), (170, 0), (550, 0), (550, 720)])

        warp_img, M, Minv = warp_image(combined_result, src, dst, (720, 720))
        searching_img = find_LR_lines(warp_img, left_line, right_line)
        w_comb_result, w_color_result = draw_lane(searching_img, left_line, right_line)

        # Drawing the lines back down onto the road
        color_result = cv2.warpPerspective(w_color_result, Minv, (c_cols, c_rows))
        comb_result = np.zeros_like(undist_img)
        comb_result[220:rows - 12, 0:cols] = color_result

        # Combine the result with the original image
        result = cv2.addWeighted(undist_img, 1, comb_result, 0.3, 0)
        cv2.imshow('result',result)

        cv2.waitKey(0)

    elif input_type == 'video': # 타입이 비디오라면
        cap = cv2.VideoCapture(input_name) # VideoCapture() 전달인자로 파일명을 넣으면 저장된 비디오를 불러온다.
        # 전달인자로 파일명이 아닌 0,1,..을 입력하면 연결된 디바이스(노트북에 달려있는 카메라 같은것 1개면 0을 입력하면 된다.)에 따라 실시간 촬영 frame을 받아 온다.
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')

        while (cap.isOpened()):
            res, frame = cap.read()
            if res == False: # 영상이 끝났을때 break
                # 저장된 파일의 목록을 출력해준다.
                fileList = os.listdir("./output")
                print("file list : {}".format(fileList))
                break
            # Correcting for Distortion
            undist_img = undistort(frame, mtx, dist)
            # resize video
            undist_img = cv2.resize(undist_img, None, fx=1 / 2, fy=1 / 2, interpolation=cv2.INTER_AREA)
            rows, cols = undist_img.shape[:2]

            combined_gradient = gradient_combine(undist_img, th_sobelx, th_sobely, th_mag, th_dir)
            #cv2.imshow('gradient combined image', combined_gradient)

            combined_hls = hls_combine(undist_img, th_h, th_l, th_s)
            #cv2.imshow('HLS combined image', combined_hls)

            combined_result = comb_result(combined_gradient, combined_hls)

            c_rows, c_cols = combined_result.shape[:2]
            s_LTop2, s_RTop2 = [c_cols / 2 - 24, 5], [c_cols / 2 + 24, 5]
            s_LBot2, s_RBot2 = [110, c_rows], [c_cols - 110, c_rows]

            src = np.float32([s_LBot2, s_LTop2, s_RTop2, s_RBot2])
            dst = np.float32([(170, 720), (170, 0), (550, 0), (550, 720)])

            warp_img, M, Minv = warp_image(combined_result, src, dst, (720, 720))
            #cv2.imshow('warp', warp_img)

            searching_img = find_LR_lines(warp_img, left_line, right_line)
            #cv2.imshow('LR searching', searching_img)

            w_comb_result, w_color_result = draw_lane(searching_img, left_line, right_line)
            #cv2.imshow('w_comb_result', w_comb_result)

            # Drawing the lines back down onto the road
            color_result = cv2.warpPerspective(w_color_result, Minv, (c_cols, c_rows))
            lane_color = np.zeros_like(undist_img)
            lane_color[220:rows - 12, 0:cols] = color_result

            # Combine the result with the original image
            result = cv2.addWeighted(undist_img, 1, lane_color, 0.3, 0)
            #cv2.imshow('result', result.astype(np.uint8))

            info, info2 = np.zeros_like(result),  np.zeros_like(result)
            info[5:110, 5:190] = (255, 255, 255)
            info2[5:110, cols-111:cols-6] = (255, 255, 255)
            info = cv2.addWeighted(result, 1, info, 0.2, 0)
            info2 = cv2.addWeighted(info, 1, info2, 0.2, 0)
            road_map = print_road_map(w_color_result, left_line, right_line)
            info2[10:105, cols-106:cols-11] = road_map
            info2 = print_road_status(info2, left_line, right_line)
            cv2.imshow('road info', info2)


            # 동영상 저장

            # 디렉토리 생성
            try:
                if not(os.path.isdir("output/")):
                    os.makedirs(os.path.join("output/"))
            except OSError as e:
                if e.errno != errno.EEXIST:
                    print("Failed to create directory!")

            frameCnt += 1 # 프레임수 카운트

            if frameCnt % 297 == 1:
                # 파일저장 번호증가와 프레임개수를 1로 초기화
                saveCnt += 1
                frameCnt = 1

                fileName = datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S") # 파일명을 시간으로 정하기
                info2Video = cv2.VideoWriter("output/info2_" + str(fileName) + ".mp4", fourcc, 29.7,(info2.shape[1], info2.shape[0]),True) # info2 결과영상 저장
                combinedVideo = cv2.VideoWriter("output/comvined_" + str(fileName) + ".mp4", fourcc, 29.7,(combined_result.shape[1], combined_result.shape[0]),False) # 차선인식 영상 저장


                # 디스크 파일 시스템 정보
                st = os.statvfs("/")
                # 총, 남은 디스크 용량 계산
                total = st.f_blocks * st.f_frsize
                used = (st.f_blocks - st.f_bfree) * st.f_frsize
                free = st.f_bavail * st.f_frsize
                # GB 단위로 출력
                print(str(saveCnt) + "번째 저장")
                print("disk total :" + str(total / 1024 / 1024 / 1024)[0:5] + "GB")
                print("disk used : " + str(used / 1024 / 1024 / 1024)[0:5] + "GB")
                print("disk free : " + str(free / 1024 / 1024 / 1024)[0:5] + "GB")
            info2Video.write(info2) # info2 결과frame을 info2Vidoe객체에 하나씩 추가? 하는건가
            combinedVideo.write(combined_result) #



            # q를 눌렀을때 종료
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cv2.waitKey(0)
                fileList = os.listdir("./output")
                print("file list : {}".format(fileList))
                break
            # if cv2.waitKey(1) & 0xFF == ord('r'):
            #     cv2.imwrite('check1.jpg', undist_img)
            #if cv2.waitKey(1) & 0xFF == ord('q'):

        info2Video.release()
        combinedVideo.release()
        cap.release()
        cv2.destroyAllWindows()

