import numpy as np
import cv2
import json
import os
from datetime import datetime
import time
from ForwardController import BraccioRobot
from InverseCalculate import BraccioInverseKinematics


class HandEyeCalibration:
    """
    基于OpenCV的手眼标定类
    用于将世界坐标系B转换到机械臂基座坐标系C
    """
    
    def __init__(self, camera_index=0, com_port='COM6'):
        """
        初始化手眼标定
        
        参数:
            camera_index (int): 摄像头索引
            com_port (str): 机械臂串口号
        """
        self.camera_index = camera_index
        
        # 初始化机械臂控制器
        self.robot = BraccioRobot(com_port=com_port)
        self.ik_solver = BraccioInverseKinematics()
        
        # 标定数据存储
        self.robot_poses = []      # 机械臂末端位姿 (4x4齐次变换矩阵)
        self.camera_observations = []  # 相机观测到的gripper位置
        
        # 标定结果
        self.hand_eye_matrix = None  # 手眼标定矩阵
        
        # 红色色块检测参数 (HSV颜色空间)
        self.red_lower1 = np.array([0, 50, 50])
        self.red_upper1 = np.array([10, 255, 255])
        self.red_lower2 = np.array([170, 50, 50])
        self.red_upper2 = np.array([180, 255, 255])
        
        # 创建保存目录
        self.save_dir = "hand_eye_calibration"
        os.makedirs(self.save_dir, exist_ok=True)
        
        print("手眼标定程序初始化完成")
        print(f"数据保存目录: {self.save_dir}")
    
    def detect_red_marker(self, image):
        """
        检测图像中的红色色块标记
        
        参数:
            image: 输入图像
        返回:
            (center_x, center_y): 红色标记中心点像素坐标，如果未检测到则返回None
        """
        # 转换到HSV颜色空间
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # 创建红色掩码（两个范围）
        mask1 = cv2.inRange(hsv, self.red_lower1, self.red_upper1)
        mask2 = cv2.inRange(hsv, self.red_lower2, self.red_upper2)
        mask = cv2.bitwise_or(mask1, mask2)
        
        # 形态学操作去除噪声
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # 查找轮廓
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # 找到最大的轮廓
            largest_contour = max(contours, key=cv2.contourArea)
            
            # 计算面积，过滤太小的区域
            area = cv2.contourArea(largest_contour)
            if area > 100:  # 最小面积阈值
                # 计算质心
                M = cv2.moments(largest_contour)
                if M["m00"] != 0:
                    center_x = int(M["m10"] / M["m00"])
                    center_y = int(M["m01"] / M["m00"])
                    return center_x, center_y, largest_contour
        
        return None
    
    def visualize_detection(self, image, detection_result):
        """
        可视化检测结果
        
        参数:
            image: 原始图像
            detection_result: 检测结果
        返回:
            可视化图像
        """
        vis_image = image.copy()
        
        if detection_result:
            center_x, center_y, contour = detection_result
            
            # 绘制轮廓
            cv2.drawContours(vis_image, [contour], -1, (0, 255, 0), 2)
            
            # 绘制中心点
            cv2.circle(vis_image, (center_x, center_y), 5, (0, 0, 255), -1)
            
            # 显示坐标
            cv2.putText(vis_image, f"({center_x}, {center_y})", 
                       (center_x + 10, center_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # 显示检测状态
            cv2.putText(vis_image, "Red marker detected", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(vis_image, "No red marker detected", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        return vis_image
    
    def get_robot_end_effector_pose(self, joint_angles):
        """
        根据关节角度计算末端执行器的位姿矩阵
        
        参数:
            joint_angles: [base, shoulder, elbow, wrist, twist] 关节角度列表
        返回:
            4x4齐次变换矩阵
        """
        # 使用正运动学计算末端执行器位置
        transformation_matrix = self.robot.calculate_forward_kinematics(*joint_angles)
        
        # 转换为numpy数组
        T = np.array(transformation_matrix, dtype=float)
        
        return T
    
    def collect_calibration_data(self):
        """
        自动收集手眼标定数据，使用预设的8个机械臂位置
        """
        print(f"\n=== 手眼标定数据收集 ===")
        print(f"机械臂将自动移动到8个预设位置，每个位置记录：")
        print(f"1. 机械臂末端执行器的位姿")
        print(f"2. 相机观测到的红色标记位置")
        print(f"\n操作说明:")
        print(f"- 请确保红色标记在相机视野内")
        print(f"- 每个位置自动采集一次，无需手动按键")
        
        # 你的8组预设点
        calibration_positions = [
            [0, 90, 180, 90, 0, 50, 50],
            [75, 90, 180, 90, 0, 50, 50],
            [80, 90, 180, 90, 0, 50, 50],
            [85, 90, 180, 90, 0, 50, 50],
            [90, 90, 180, 90, 0, 50, 50],
            [95, 90, 180, 90, 0, 50, 50],
            [100, 90, 180, 90, 0, 50, 50],
            [105, 90, 180, 90, 0, 50, 50],
        ]
        
        cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            print(f"错误: 无法打开摄像头 {self.camera_index}")
            return False
        
        collected_count = 0
        
        for idx, position in enumerate(calibration_positions):
            print(f"\n移动到位置 {idx + 1}: {position}")
            self.robot.move_to_angles(*position)
            time.sleep(4)  # 等待机械臂稳定
            
            # 自动采集一帧
            ret, frame = cap.read()
            if not ret:
                print("错误: 无法读取摄像头画面")
                continue
            
            detection_result = self.detect_red_marker(frame)
            if detection_result:
                center_x, center_y, _ = detection_result
                # 只取前五个关节角度用于正解
                joint_angles = position[:5]
                robot_pose = self.get_robot_end_effector_pose(joint_angles)
                self.robot_poses.append(robot_pose)
                self.camera_observations.append([center_x, center_y])
                image_filename = os.path.join(self.save_dir, f"calibration_pose_{collected_count:03d}.jpg")
                cv2.imwrite(image_filename, frame)
                print(f"已记录数据点 {collected_count+1}: 像素坐标({center_x}, {center_y})，保存图像: {image_filename}")
                collected_count += 1
            else:
                print(f"未检测到红色标记，跳过该点")
        
        cap.release()
        cv2.destroyAllWindows()
        
        if collected_count < 4:
            print(f"警告: 只收集到 {collected_count} 个数据点，少于最低要求的4个")
            return False
        
        print(f"\n数据收集完成，共收集 {collected_count} 个有效数据点")
        return True
    
    def perform_hand_eye_calibration(self):
        """
        执行手眼标定计算
        """
        if len(self.robot_poses) < 4:
            print("错误: 标定数据不足，至少需要4组数据")
            return False
        
        print(f"\n=== 开始手眼标定计算 ===")
        print(f"使用 {len(self.robot_poses)} 组数据进行标定...")
        
        try:
            # 准备数据格式
            R_gripper2base = []  # 机械臂末端到基座的旋转矩阵
            t_gripper2base = []  # 机械臂末端到基座的平移向量
            
            # 由于我们只有2D相机观测，创建虚拟的相机位姿变化
            R_target2cam = []    # 目标到相机的旋转矩阵
            t_target2cam = []    # 目标到相机的平移向量
            
            # 处理机械臂位姿数据
            for pose in self.robot_poses:
                # 提取旋转矩阵和平移向量
                R = pose[:3, :3]
                t = pose[:3, 3].reshape(-1, 1)
                
                R_gripper2base.append(R)
                t_gripper2base.append(t)
            
            # 处理相机观测数据
            # 由于相机观测是2D的，我们需要创建相对变换
            base_observation = self.camera_observations[0]
            
            for obs in self.camera_observations:
                # 计算相对于第一个观测的像素位置变化
                dx = obs[0] - base_observation[0]
                dy = obs[1] - base_observation[1]
                
                # 创建一个简化的平移变换（假设没有旋转）
                R_cam = np.eye(3, dtype=np.float64)
                t_cam = np.array([[dx], [dy], [0]], dtype=np.float64)
                
                R_target2cam.append(R_cam)
                t_target2cam.append(t_cam)
            
            # 使用OpenCV的手眼标定函数
            # 注意：这里我们使用相对变换而不是绝对位姿
            R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
                R_gripper2base[:-1], t_gripper2base[:-1],  # 相对的机械臂位姿变化
                R_target2cam[:-1], t_target2cam[:-1],      # 相对的相机观测变化
                method=cv2.CALIB_HAND_EYE_TSAI
            )
            
            # 构建完整的手眼标定矩阵
            self.hand_eye_matrix = np.eye(4)
            self.hand_eye_matrix[:3, :3] = R_cam2gripper
            self.hand_eye_matrix[:3, 3] = t_cam2gripper.flatten()
            
            print("手眼标定成功!")
            print("手眼标定矩阵 (相机到末端执行器):")
            print(self.hand_eye_matrix)
            
            # 保存标定结果
            self.save_calibration_results()
            return True
            
        except Exception as e:
            print(f"手眼标定失败: {e}")
            print("这可能是由于数据不足或位姿变化太小导致的")
            print("建议:")
            print("1. 增加更多标定位置")
            print("2. 确保机械臂在不同位置间有足够大的移动")
            print("3. 检查红色标记检测的准确性")
            return False
    
    def save_calibration_results(self):
        """
        保存手眼标定结果
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 保存为JSON格式
        calibration_file = os.path.join(self.save_dir, f"hand_eye_calibration_{timestamp}.json")
        
        calibration_data = {
            "timestamp": timestamp,
            "num_data_points": len(self.robot_poses),
            "hand_eye_matrix": self.hand_eye_matrix.tolist(),
            "camera_observations": self.camera_observations,
            "robot_poses": [pose.tolist() for pose in self.robot_poses]
        }
        
        with open(calibration_file, 'w', encoding='utf-8') as f:
            json.dump(calibration_data, f, indent=4, ensure_ascii=False)
        
        print(f"手眼标定结果已保存到: {calibration_file}")
        
        # 同时保存为numpy格式
        np_file = os.path.join(self.save_dir, f"hand_eye_matrix_{timestamp}.npz")
        np.savez(np_file,
                hand_eye_matrix=self.hand_eye_matrix,
                camera_observations=np.array(self.camera_observations),
                robot_poses=np.array(self.robot_poses))
        print(f"手眼标定矩阵已保存到: {np_file}")
    
    def load_calibration_results(self, calibration_file):
        """
        加载手眼标定结果
        
        参数:
            calibration_file (str): 标定文件路径
        """
        if calibration_file.endswith('.json'):
            with open(calibration_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self.hand_eye_matrix = np.array(data['hand_eye_matrix'])
        elif calibration_file.endswith('.npz'):
            data = np.load(calibration_file)
            self.hand_eye_matrix = data['hand_eye_matrix']
        else:
            raise ValueError("不支持的文件格式")
        
        print(f"已加载手眼标定结果: {calibration_file}")
    
    def transform_coordinates(self, camera_x, camera_y, camera_z=0):
        """
        将相机坐标转换为机械臂基座坐标
        
        参数:
            camera_x, camera_y: 相机坐标系中的坐标
            camera_z: 相机坐标系中的Z坐标（默认为0）
        返回:
            (base_x, base_y, base_z): 机械臂基座坐标系中的坐标
        """
        if self.hand_eye_matrix is None:
            print("错误: 请先进行手眼标定或加载标定结果")
            return None
        
        # 构建齐次坐标
        camera_point = np.array([camera_x, camera_y, camera_z, 1]).reshape(4, 1)
        
        # 应用手眼标定变换
        base_point = self.hand_eye_matrix @ camera_point
        
        return base_point[:3].flatten()


def main():
    """
    主函数 - 执行完整的手眼标定流程
    """
    print("=== 手眼标定程序 ===")
    print("此程序将帮助您完成手眼标定，实现世界坐标系B到机械臂基座坐标系C的转换")
    print("请确保:")
    print("1. 机械臂gripper上安装了红色标记")
    print("2. 机械臂已正确连接并可以通信")
    print("3. 相机可以清晰看到机械臂工作区域")
    
    # 根据实际情况修改串口号
    com_port = input("请输入机械臂串口号 (默认: COM6): ").strip() or "COM6"
    
    # 创建手眼标定对象
    calibrator = HandEyeCalibration(camera_index=0, com_port=com_port)
    
    # 步骤1: 收集标定数据
    print("\n步骤1: 收集手眼标定数据")
    if not calibrator.collect_calibration_data():
        print("数据收集失败")
        return
    
    # 步骤2: 执行手眼标定
    print("\n步骤2: 执行手眼标定计算")
    if not calibrator.perform_hand_eye_calibration():
        print("手眼标定失败")
        return
    
    # 关闭机械臂连接
    calibrator.robot.close_serial()
    
    print("\n手眼标定完成!")
    print("标定结果已保存到 hand_eye_calibration 目录")
    print("你现在可以使用手眼标定矩阵进行坐标转换")


if __name__ == "__main__":
    main()