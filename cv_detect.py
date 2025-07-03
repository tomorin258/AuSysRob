import cv2
import numpy as np
import time
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from PositionCalculate import PositionCalculator
from HandEyeCalibration import HandEyeCalibrator
from InverseCalculate import BraccioInverseKinematics
from ForwardController import BraccioRobot

# 配置参数
CAMERA_ID = 0
CAMERA_CALIBRATION_FILE = "camera_calibration.pkl"
HOMOGRAPHY_MATRIX_FILE = "homography_matrix.pkl"
HAND_EYE_CALIBRATION_FILE = "hand_eye_calibration.pkl"
BRACCIO_COM_PORT = 'COM8'
BRACCIO_BAUD_RATE = 115200
BRACCIO_TIMEOUT = 5
LEGO_HEIGHT_MM = 20

# 图像预处理与红色识别

def preprocess_and_detect_red(frame):
    # 去噪（高斯模糊）
    blurred = cv2.GaussianBlur(frame, (7, 7), 0)
    # 增强对比度（自适应直方图均衡）
    lab = cv2.cvtColor(blurred, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    # 灰度化
    gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
    # 红色掩码（在增强后原图上做）
    hsv = cv2.cvtColor(enhanced, cv2.COLOR_BGR2HSV)
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([179, 255, 255])
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = mask1 | mask2
    # 形态学去噪
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5,5), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7,7), np.uint8))
    # 找最大轮廓
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        center_x = x + w // 2
        center_y = y + h // 2
        area = cv2.contourArea(c)
        return (center_x, center_y), (w, h), area, mask, gray, enhanced
    return None, None, None, mask, gray, enhanced

def main():
    # 加载标定
    position_calculator = PositionCalculator(
        camera_id=CAMERA_ID,
        model_path=None,
        calibration_file=CAMERA_CALIBRATION_FILE,
        homography_file=HOMOGRAPHY_MATRIX_FILE
    )
    hand_eye_calibrator = HandEyeCalibrator(
        calibration_file=HAND_EYE_CALIBRATION_FILE,
        lego_height_mm=LEGO_HEIGHT_MM
    )
    inverse_kinematics_solver = BraccioInverseKinematics()
    robot = BraccioRobot(
        com_port=BRACCIO_COM_PORT,
        baud_rate=BRACCIO_BAUD_RATE,
        timeout=BRACCIO_TIMEOUT
    )
    if robot.s is None or not robot.s.is_open:
        print("机械臂串口连接失败。"); return
    robot.move_to_home()
    time.sleep(2)
    cap = cv2.VideoCapture(CAMERA_ID)
    if not cap.isOpened():
        print("无法打开摄像头"); return
    try:
        stable_count = 0
        last_center = None
        STABLE_THRESHOLD = 10  # 连续检测到多少帧认为稳定
        while True:
            ret, frame = cap.read()
            if not ret:
                print("无法读取摄像头帧"); break
            result = preprocess_and_detect_red(frame)
            center, wh, area, mask, gray, enhanced = result
            show_frame = frame.copy()
            if center and wh:
                w, h = wh
                cv2.rectangle(show_frame, (center[0]-w//2, center[1]-h//2), (center[0]+w//2, center[1]+h//2), (0,255,0), 2)
                cv2.circle(show_frame, center, 5, (255,0,0), -1)
                # 判断色块是否稳定
                if last_center and np.linalg.norm(np.array(center) - np.array(last_center)) < 5:
                    stable_count += 1
                else:
                    stable_count = 1
                last_center = center
                if stable_count >= STABLE_THRESHOLD:
                    # 坐标转换与抓取逻辑
                    pixel_coords = np.array([[[center[0], center[1]]]], dtype=np.float32)
                    world_coords_2d = cv2.perspectiveTransform(pixel_coords, position_calculator.homography_matrix)[0][0]
                    world_x, world_y = world_coords_2d
                    world_z = LEGO_HEIGHT_MM
                    block_coords_C = hand_eye_calibrator.transform_point_B_to_C([world_x, world_y, world_z])
                    if block_coords_C is not None:
                        target_x, target_y, target_z = block_coords_C
                        print(f"检测到红色色块，机械臂基座坐标: ({target_x:.2f}, {target_y:.2f}, {target_z:.2f}) mm")
                        gripper_pos = min(max(int(w * 0.7), 30), 120)
                        twist = 90
                        joint_angles_deg = inverse_kinematics_solver.calculate_joint_angles(target_x, target_y, target_z)
                        if joint_angles_deg:
                            base, shoulder, elbow, wrist, _ = [int(angle) for angle in joint_angles_deg]
                            print(f"关节角度: Base={base}, Shoulder={shoulder}, Elbow={elbow}, Wrist={wrist}, Twist={twist}, Gripper={gripper_pos}")
                            robot.move_to_angles(base, shoulder, elbow, wrist, twist, gripper_pos=gripper_pos, move_time=100)
                            time.sleep(2)
                            robot.move_to_angles(base, shoulder, elbow, wrist, twist, gripper_pos=30, move_time=50)
                            time.sleep(2)
                            robot.move_to_angles(base, shoulder, elbow+10, wrist, twist, gripper_pos=30, move_time=100)
                            time.sleep(2)
                            robot.move_to_home()
                            time.sleep(2)
                            robot.move_to_angles(0, 90, 180, 90, 0, gripper_pos=120, move_time=50)
                            time.sleep(2)
                        else:
                            print("目标点不可达，无法计算关节角度。请调整色块位置。")
                    else:
                        print("坐标转换失败。")
                    stable_count = 0  # 抓取后重置
            else:
                stable_count = 0
                last_center = None
                # print("未检测到红色色块。请确保色块在摄像头视野内。")
            cv2.imshow("Red Block Detection", show_frame)
            cv2.imshow("Red Mask", mask)
            cv2.imshow("Gray", gray)
            cv2.imshow("Enhanced", enhanced)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("用户选择退出。"); break
    except KeyboardInterrupt:
        print("\n程序被用户中断。")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        robot.close_serial()
        print("程序结束。")

if __name__ == "__main__":
    main()