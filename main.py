import os
import sys
import time

# 确保项目根目录在 Python 路径中，以便正确导入模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from PositionCalculate import PositionCalculator
from HandEyeCalibration import HandEyeCalibrator
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

def main():
    print("--- 视觉引导机械臂抓取系统启动程序 ---")
    print("请按照提示一步步完成校准和测试。")

    # 1. 初始化 PositionCalculator (相机内参标定与平面校准)
    print_step_header(1, "初始化相机和平面校准模块")
    position_calculator = PositionCalculator(
        camera_id=CAMERA_ID,
        model_path=YOLO_MODEL_PATH,
        calibration_file=CAMERA_CALIBRATION_FILE,
        homography_file=HOMOGRAPHY_MATRIX_FILE
    )

    # # 1.1 相机内参标定
    # if not (position_calculator.camera_matrix is not None and position_calculator.dist_coeffs is not None):
    #     print_step_header("1.1", "相机内参标定 (A -> B 的部分映射)")
    #     print("请准备棋盘格，并按照程序提示进行相机内参标定。")
    #     input("按 Enter 键开始相机内参标定...")
    #     position_calculator.calibrate_camera(
    #         checkerboard_size=CHECKERBOARD_SIZE,
    #         square_size_mm=SQUARE_SIZE_MM,
    #         num_images=NUM_CALIBRATION_IMAGES
    #     )
    #     if not (position_calculator.camera_matrix is not None and position_calculator.dist_coeffs is not None):
    #         print("错误: 相机内参标定失败或用户中断。请解决问题后重新运行程序。")
    #         return

    # # 1.2 平面校准 (单应性矩阵计算)
    # if position_calculator.homography_matrix is None:
    #     print_step_header("1.2", "平面校准 (A -> B 的单应性矩阵计算)")
    #     print("请将棋盘格平放于工作平面上，并按照程序提示进行平面校准。")
    #     input("按 Enter 键开始平面校准...")
    #     position_calculator.calculate_plane_homography(
    #         checkerboard_size=CHECKERBOARD_SIZE,
    #         square_size_mm=SQUARE_SIZE_MM
    #     )
    #     if position_calculator.homography_matrix is None:
    #         print("错误: 平面校准失败或用户中断。请解决问题后重新运行程序。")
    #         return

    # 2. 初始化 HandEyeCalibrator (手眼标定)
    print_step_header(2, "初始化手眼标定模块")
    hand_eye_calibrator = HandEyeCalibrator(
        calibration_file=HAND_EYE_CALIBRATION_FILE,
        lego_height_mm=LEGO_HEIGHT_MM
    )

    # # 2.1 手眼标定 (B -> C 的映射)
    # if hand_eye_calibrator.transform_matrix is None:
    #     print_step_header("2.1", "手眼标定 (B -> C 的转换矩阵计算)")
    #     print("此步骤需要您手动移动机械臂并输入其基座坐标系下的位置。")
    #     print(f"您需要收集 {NUM_HAND_EYE_POINTS} 组点对。")
    #     input("按 Enter 键开始手眼标定数据收集...")
    #     hand_eye_calibrator.collect_and_calibrate(
    #         position_calculator=position_calculator,
    #         num_points=NUM_HAND_EYE_POINTS
    #     )
    #     if hand_eye_calibrator.transform_matrix is None:
    #         print("错误: 手眼标定失败或用户中断。请解决问题后重新运行程序。")
    #         return

    # 3. 初始化 BraccioInverseKinematics (逆向运动学)
    print_step_header(3, "初始化逆向运动学模块")
    inverse_kinematics_solver = BraccioInverseKinematics()
    print("逆向运动学模块初始化完成。")

    # 4. 初始化 BraccioRobot (机械臂控制)
    print_step_header(4, "初始化机械臂控制模块")
    robot = BraccioRobot(
        com_port=BRACCIO_COM_PORT,
        baud_rate=BRACCIO_BAUD_RATE,
        timeout=BRACCIO_TIMEOUT
    )
    if robot.s is None or not robot.s.is_open:
        print("错误: 机械臂串口连接失败。请检查串口号或连接。")
        return

    # 移动机械臂到 Home 位置
    print_step_header("4.1", "移动机械臂到 Home 位置")
    robot.move_to_home()
    time.sleep(2) # 等待机械臂到位

    # 5. 整合测试：实时目标检测、坐标转换、逆向运动学和机械臂抓取
    print_step_header(5, "系统整合测试：实时抓取")
    print("现在系统将尝试实时检测乐高积木，并引导机械臂进行抓取。")
    print("请确保乐高积木在摄像头视野内。按 'q' 退出。")

    try:
        while True:
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

                    print("Open the gripper...")
                    robot.move_to_angles(0, 90, 180, 90, 0, gripper_pos=90, move_time=100) # 夹持器打开
                    time.sleep(2) # 等待机械臂到位

                    # 移动机械臂到目标位置并抓取 (这里示例夹持器打开，然后关闭抓取)
                    print("移动机械臂到目标位置...")
                    robot.move_to_angles(base, shoulder, elbow, wrist, twist, gripper_pos=90, move_time=100) # 夹持器打开
                    time.sleep(2) # 等待机械臂到位

                    print("执行抓取动作...")
                    robot.move_to_angles(base, shoulder, elbow, wrist, twist, gripper_pos=0, move_time=50) # 夹持器关闭
                    time.sleep(2)

                    # 抓取后抬起机械臂 (可以根据需要调整)
                    print("抓取后抬起...")
                    robot.move_to_angles(base, shoulder, elbow+10, 90, twist, gripper_pos=0, move_time=100) # 提高腕部，保持夹持器关闭
                    time.sleep(2)

                    # 移动到安全位置或放置区 (这里示例回到 Home)
                    print("移动到 Home 位置...")
                    robot.move_to_home()
                    time.sleep(2)

                    # 放置后打开夹持器 (示例)
                    print("放置后打开夹持器...")
                    robot.move_to_angles(0, 90, 180, 90, 0, gripper_pos=90, move_time=50)
                    time.sleep(2)
                    
                else:
                    print("目标点不可达，无法计算关节角度。请调整乐高积木位置。")
                    time.sleep(1) # 短暂等待，避免频繁尝试
            else:
                print("未检测到乐高积木。请确保积木在摄像头视野内。")
                time.sleep(1) # 短暂等待，避免频繁检测

            # 检查用户是否退出
            key_input = input("按 Enter 继续，按 'q' 退出测试...").strip().lower()
            if key_input == 'q':
                print("用户选择退出测试。")
                break

    except KeyboardInterrupt:
        print("\n程序被用户中断。")
    finally:
        # 关闭串口连接
        robot.close_serial()
        print("程序结束。")

if __name__ == "__main__":
    main() 