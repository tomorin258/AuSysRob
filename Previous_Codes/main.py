import os
import sys
import time
import cv2
import numpy as np

# 确保项目根目录在 Python 路径中，以便正确导入模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from Previous_Codes.PositionCalculate import PositionCalculator
from Previous_Codes.HandEyeCalibration import HandEyeCalibrator
from InverseCalculate import BraccioInverseKinematics
from ForwardController import BraccioRobot

# --- 配置参数 ---
CAMERA_ID = 0  # 摄像头ID，通常是0或1
YOLO_MODEL_PATH = "yolo11n.pt"  # YOLO模型文件路径
CAMERA_CALIBRATION_FILE = "camera_calibration.pkl"  # 相机内参标定数据保存文件
HOMOGRAPHY_MATRIX_FILE = "homography_matrix.pkl"  # 单应性矩阵数据保存文件
HAND_EYE_CALIBRATION_FILE = "hand_eye_calibration.pkl"  # 手眼标定数据保存文件
BRACCIO_COM_PORT = 'COM8'  # 机械臂串口号，根据您的系统进行修改 (例如 '/dev/ttyUSB0' 或 'COMx')
BRACCIO_BAUD_RATE = 115200
BRACCIO_TIMEOUT = 5

# 相机内参标定和平面校准的棋盘格参数
CHECKERBOARD_SIZE = (9, 6)  # 棋盘格内角点数量 (cols, rows)
SQUARE_SIZE_MM = 25  # 棋盘格每个方块的物理边长，单位毫米
NUM_CALIBRATION_IMAGES = 15  # 相机内参标定所需图片数量

# 手眼标定参数
NUM_HAND_EYE_POINTS = 5  # 手眼标定所需点对数量 (至少3个，建议5个或更多)
LEGO_HEIGHT_MM = 20  # 乐高积木的固定高度，单位毫米

def print_step_header(step_num, title):
    print("\n" + "="*50)
    print(f"步骤 {step_num}: {title}")
    print("="*50)

def detect_red_block(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([179, 255, 255])
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = mask1 | mask2
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5,5), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        center_x = x + w // 2
        center_y = y + h // 2
        area = cv2.contourArea(c)
        return (center_x, center_y), (w, h), area, mask
    return None, None, None, mask

def main():
    print("--- 视觉引导机械臂抓取系统启动程序 ---")
    print("请按照提示一步步完成校准和测试。")

    # 1. 相机内参标定和平面校准
    print_step_header(1, "相机内参标定和平面校准")
    position_calculator = PositionCalculator(
        camera_id=CAMERA_ID,
        model_path=YOLO_MODEL_PATH,
        calibration_file=CAMERA_CALIBRATION_FILE,
        homography_file=HOMOGRAPHY_MATRIX_FILE
    )
    if not (position_calculator.camera_matrix is not None and position_calculator.dist_coeffs is not None):
        print("请准备棋盘格，按 Enter 开始相机内参标定...")
        input()
        position_calculator.calibrate_camera(
            checkerboard_size=CHECKERBOARD_SIZE,
            square_size_mm=SQUARE_SIZE_MM,
            num_images=NUM_CALIBRATION_IMAGES
        )
        if not (position_calculator.camera_matrix is not None and position_calculator.dist_coeffs is not None):
            print("错误: 相机内参标定失败或用户中断。请解决问题后重新运行程序。")
            return
    if position_calculator.homography_matrix is None:
        print("请将棋盘格平放于工作平面上，按 Enter 开始平面校准...")
        input()
        position_calculator.calculate_plane_homography(
            checkerboard_size=CHECKERBOARD_SIZE,
            square_size_mm=SQUARE_SIZE_MM
        )
        if position_calculator.homography_matrix is None:
            print("错误: 平面校准失败或用户中断。请解决问题后重新运行程序。")
            return

    # 2. 手眼标定
    print_step_header(2, "手眼标定 (世界坐标系->机械臂坐标系)")
    hand_eye_calibrator = HandEyeCalibrator(
        calibration_file=HAND_EYE_CALIBRATION_FILE,
        lego_height_mm=LEGO_HEIGHT_MM
    )
    if hand_eye_calibrator.transform_matrix is None:
        print(f"需要收集 {NUM_HAND_EYE_POINTS} 组点对。按 Enter 开始手眼标定...")
        input()
        hand_eye_calibrator.collect_and_calibrate(
            position_calculator=position_calculator,
            num_points=NUM_HAND_EYE_POINTS
        )
        if hand_eye_calibrator.transform_matrix is None:
            print("错误: 手眼标定失败或用户中断。请解决问题后重新运行程序。")
            return

    # 3. 初始化逆向运动学和机械臂
    print_step_header(3, "初始化逆向运动学和机械臂")
    inverse_kinematics_solver = BraccioInverseKinematics()
    robot = BraccioRobot(
        com_port=BRACCIO_COM_PORT,
        baud_rate=BRACCIO_BAUD_RATE,
        timeout=BRACCIO_TIMEOUT
    )
    if robot.s is None or not robot.s.is_open:
        print("错误: 机械臂串口连接失败。请检查串口号或连接。")
        return
    robot.move_to_home()
    time.sleep(2)

    # 4. 实时色块识别与抓取
    print_step_header(4, "实时红色色块识别与抓取")
    cap = cv2.VideoCapture(CAMERA_ID)
    if not cap.isOpened():
        print("无法打开摄像头")
        return
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("无法读取摄像头帧")
                break
            center, (w, h), area, mask = detect_red_block(frame)
            show_frame = frame.copy()
            if center:
                cv2.rectangle(show_frame, (center[0]-w//2, center[1]-h//2), (center[0]+w//2, center[1]+h//2), (0,255,0), 2)
                cv2.circle(show_frame, center, 5, (255,0,0), -1)
                # 坐标转换
                pixel_coords = np.array([[[center[0], center[1]]]], dtype=np.float32)
                world_coords_2d = cv2.perspectiveTransform(pixel_coords, position_calculator.homography_matrix)[0][0]
                world_x, world_y = world_coords_2d
                world_z = LEGO_HEIGHT_MM
                # 机械臂坐标转换
                block_coords_C = hand_eye_calibrator.transform_point_B_to_C([world_x, world_y, world_z])
                if block_coords_C is not None:
                    target_x, target_y, target_z = block_coords_C
                    print(f"检测到红色色块，机械臂基座坐标: ({target_x:.2f}, {target_y:.2f}, {target_z:.2f}) mm")
                    # 动态调整夹爪参数
                    gripper_pos = min(max(int(w * 0.7), 30), 120)  # 你可根据实际调整映射关系
                    twist = 90  # 如需根据色块角度调整可扩展
                    # 逆向运动学
                    joint_angles_deg = inverse_kinematics_solver.calculate_joint_angles(target_x, target_y, target_z)
                    if joint_angles_deg:
                        base, shoulder, elbow, wrist, _ = [int(angle) for angle in joint_angles_deg]
                        print(f"关节角度: Base={base}, Shoulder={shoulder}, Elbow={elbow}, Wrist={wrist}, Twist={twist}, Gripper={gripper_pos}")
                        # 移动机械臂
                        robot.move_to_angles(base, shoulder, elbow, wrist, twist, gripper_pos=gripper_pos, move_time=100)
                        time.sleep(2)
                        # 抓取
                        robot.move_to_angles(base, shoulder, elbow, wrist, twist, gripper_pos=30, move_time=50)
                        time.sleep(2)
                        # 抬起
                        robot.move_to_angles(base, shoulder, elbow+10, wrist, twist, gripper_pos=30, move_time=100)
                        time.sleep(2)
                        # 回home
                        robot.move_to_home()
                        time.sleep(2)
                        # 放置后张开夹爪
                        robot.move_to_angles(0, 90, 180, 90, 0, gripper_pos=120, move_time=50)
                        time.sleep(2)
                    else:
                        print("目标点不可达，无法计算关节角度。请调整色块位置。")
            # 获取乐高积木在机械臂基座坐标系C下的三维坐标
            block_coords_C = hand_eye_calibrator.get_block_coordinate_in_robot_base(position_calculator)

            if block_coords_C:
                target_x, target_y, target_z = block_coords_C
                print(f"检测到乐高积木在机械臂基座坐标系C下的坐标: ({target_x:.2f}, {target_y:.2f}, {target_z:.2f}) mm")

                # 计算关节角度
                joint_angles_deg = inverse_kinematics_solver.calculate_joint_angles(target_x, target_y, target_z)

                if joint_angles_deg:
                    base, shoulder, elbow, wrist, twist = [int(angle) for angle in joint_angles_deg]
                    print(f"计算出的关节角度: Base={base}, Shoulder={shoulder}, Elbow={elbow}, Wrist={wrist}, Twist={twist}")

                    # 1. 夹持器先打开
                    print("Open the gripper...")
                    robot.move_to_angles(0, 90, 180, 90, 0, gripper_pos=90, move_time=100)
                    time.sleep(2)

                    # 2. 移动到积木位置
                    print("移动机械臂到目标位置...")
                    robot.move_to_angles(base, shoulder, elbow, wrist, twist, gripper_pos=90, move_time=100)
                    time.sleep(2)

                    # 3. 合拢夹持器抓取
                    print("执行抓取动作...")
                    robot.move_to_angles(base, shoulder, elbow, wrist, twist, gripper_pos=0, move_time=50)
                    time.sleep(2)

                    # 4. 抓取后抬起elbow
                    print("抓取后抬起...")
                    robot.move_to_angles(base, shoulder, elbow+10, wrist, twist, gripper_pos=0, move_time=100)
                    time.sleep(2)

                    # 5. 回home位置，保持夹持器关闭
                    print("移动到 Home 位置...")
                    robot.move_to_home()
                    time.sleep(2)

                    # 放置后打开夹持器 (示例)
                    print("放置后打开夹持器...")
                    robot.move_to_angles(0, 90, 180, 90, 0, gripper_pos=90, move_time=50)
                    time.sleep(2)
                    
                else:
                    print("坐标转换失败。")
            else:
                print("未检测到红色色块。请确保色块在摄像头视野内。")
            cv2.imshow("Red Block Detection", show_frame)
            cv2.imshow("Red Mask", mask)
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