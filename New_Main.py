import cv2
import numpy as np
import os
import json
import time
from ForwardController import BraccioRobot
from InverseCalculate import BraccioInverseKinematics
from New_CameraCalibration import CameraCalibration
from New_HandEyeCalibration import HandEyeCalibration

def load_latest_file(directory, prefix, ext):
    """加载指定目录下最新的标定参数文件"""
    files = [f for f in os.listdir(directory) if f.startswith(prefix) and f.endswith(ext)]
    if not files:
        raise FileNotFoundError(f"{directory} 下未找到 {prefix} 开头的 {ext} 文件")
    files.sort()
    return os.path.join(directory, files[-1])

def detect_lego_block(frame):
    """
    这里以红色色块为例，实际可替换为你的积木检测算法
    返回像素坐标(x, y)
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower = np.array([0, 100, 100])
    upper = np.array([10, 255, 255])
    mask1 = cv2.inRange(hsv, lower, upper)
    lower2 = np.array([160, 100, 100])
    upper2 = np.array([180, 255, 255])
    mask2 = cv2.inRange(hsv, lower2, upper2)
    mask = cv2.bitwise_or(mask1, mask2)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        c = max(contours, key=cv2.contourArea)
        M = cv2.moments(c)
        if M["m00"] > 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            return (cx, cy)
    return None

def main():
    # 1. 加载标定参数
    cam_calib_file = load_latest_file("calibration_data", "camera_calibration", ".json")
    handeye_file = load_latest_file("hand_eye_calibration", "hand_eye_calibration", ".json")
    print(f"加载相机标定参数: {cam_calib_file}")
    print(f"加载手眼标定参数: {handeye_file}")

    # 加载相机内参
    cam_calib = CameraCalibration()
    cam_calib.load_calibration_results(cam_calib_file)
    # 加载手眼标定
    handeye = HandEyeCalibration()
    handeye.load_calibration_results(handeye_file)

    # 初始化机械臂
    robot = BraccioRobot(com_port='COM6', baud_rate=115200, timeout=5)
    ik_solver = BraccioInverseKinematics()

    # 机械臂初始位置
    home_angles = [0, 90, 180, 90, 0]
    robot.move_to_angles(*home_angles, gripper_pos=50, move_time=50)
    time.sleep(2)

    # 打开摄像头
    cap = cv2.VideoCapture(0)
    print("自动抓取模式启动，按q退出。")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("摄像头读取失败")
            break

        lego_pixel = detect_lego_block(frame)
        if lego_pixel:
            cv2.circle(frame, lego_pixel, 8, (0,255,0), -1)
            cv2.putText(frame, f"Block: {lego_pixel}", (lego_pixel[0]+10, lego_pixel[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
            cv2.imshow("Detect", frame)
            print("检测到积木，开始自动抓取流程...")

            # 1. 像素坐标->世界坐标B
            world_x, world_y = cam_calib.pixel_to_world_coordinates(lego_pixel[0], lego_pixel[1], world_z=0)
            # 2. 世界坐标B->机械臂基座坐标C
            base_xyz = handeye.transform_coordinates(world_x, world_y, camera_z=0)
            # 3. 固定Z高度
            z_grab = 20.0  # mm，积木高度
            z_above = z_grab + 30.0  # mm，上方30mm
            base_xyz_above = base_xyz.copy()
            base_xyz_above[2] = z_above
            base_xyz[2] = z_grab

            # 4. 逆解
            joint_angles_above = ik_solver.calculate_joint_angles(base_xyz_above[0], base_xyz_above[1], base_xyz_above[2])
            joint_angles_grab = ik_solver.calculate_joint_angles(base_xyz[0], base_xyz[1], base_xyz[2])
            if not joint_angles_above or not joint_angles_grab:
                print("目标点不可达，跳过")
                continue

            # 5. 抓取动作流程
            # 5.1 抬高到上方
            robot.move_to_angles(*[int(a) for a in joint_angles_above], gripper_pos=100, move_time=50)  # 夹爪张开
            time.sleep(1.2)
            # 5.2 下到抓取高度
            robot.move_to_angles(*[int(a) for a in joint_angles_grab], gripper_pos=100, move_time=50)
            time.sleep(1.2)
            # 5.3 合上夹爪
            robot.move_to_angles(*[int(a) for a in joint_angles_grab], gripper_pos=30, move_time=50)
            time.sleep(1.2)
            # 5.4 再抬高
            robot.move_to_angles(*[int(a) for a in joint_angles_above], gripper_pos=30, move_time=50)
            time.sleep(1.2)
            # 5.5 回到初始位置
            robot.move_to_angles(*home_angles, gripper_pos=30, move_time=50)
            time.sleep(1.2)
            # 5.6 松开夹爪
            robot.move_to_angles(*home_angles, gripper_pos=100, move_time=50)
            time.sleep(1.2)

            print("本次抓取完成，等待下一个目标...")

            # 等待积木被移走，避免重复抓取同一个
            time.sleep(2)
            continue

        cv2.imshow("Detect", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    robot.close_serial()

if __name__ == "__main__":
    main()