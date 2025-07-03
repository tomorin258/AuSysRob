import numpy as np
import cv2
import json
import os
from datetime import datetime
import time
from ForwardController import BraccioRobot
from InverseCalculate import BraccioInverseKinematics
from New_CameraCalibration import CameraCalibration
import pickle


class HandEyeCalibration:
    """
    基于OpenCV的手眼标定类
    用于将世界坐标系B转换到机械臂基座坐标系C
    """
    
    def __init__(self, camera_index=0, com_port='COM6', cam_calib_instance=None):
        """
        初始化手眼标定
        
        参数:
            camera_index (int): 摄像头索引
            com_port (str): 机械臂串口号
            cam_calib_instance (CameraCalibration): 已初始化并加载相机参数的 CameraCalibration 实例
        """
        self.camera_index = camera_index
        self.cam_calib = cam_calib_instance
        
        # 初始化机械臂控制器
        self.robot = BraccioRobot(com_port=com_port)
        self.ik_solver = BraccioInverseKinematics()
        
        # 标定数据存储
        self.robot_poses = []      # 机械臂末端位姿 (4x4齐次变换矩阵，或其平移向量)
        self.camera_observations = []  # 相机观测到的在世界坐标系B下的点 (x, y, 0)
        
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
        if self.cam_calib is None or not self.cam_calib.camera_matrix_loaded:
            print("错误: 未提供有效的 CameraCalibration 实例或相机参数未加载。无法进行像素到世界坐标的转换。")
            return False

        print(f"\n=== 手眼标定数据收集 ===")
        print(f"机械臂将自动移动到8个预设位置，每个位置记录：")
        print(f"1. 机械臂末端执行器的位姿")
        print(f"2. 相机观测到的红色标记在世界坐标系B下的位置")
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
                
                # 将像素坐标转换为世界坐标系B下的点 (Z=0)
                world_x, world_y = self.cam_calib.pixel_to_world_coordinates(center_x, center_y, world_z=0)
                
                # 只取前五个关节角度用于正解
                joint_angles = position[:5]
                robot_pose = self.get_robot_end_effector_pose(joint_angles)
                self.robot_poses.append(robot_pose)
                self.camera_observations.append([world_x, world_y, 0.0])
                image_filename = os.path.join(self.save_dir, f"calibration_pose_{collected_count:03d}.jpg")
                cv2.imwrite(image_filename, frame)
                print(f"已记录数据点 {collected_count+1}: 世界坐标B({world_x:.2f}, {world_y:.2f}, 0.0) mm，保存图像: {image_filename}")
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
        执行手眼标定。
        返回:
            bool: 如果标定成功则返回 True，否则返回 False。
        """
        print(f"DEBUG: Starting perform_hand_eye_calibration.")
        print(f"DEBUG: len(self.robot_poses) before np.array: {len(self.robot_poses)}")
        print(f"DEBUG: len(self.camera_observations) before np.array: {len(self.camera_observations)}")

        # 将列表转换为Numpy数组
        robot_poses_np = np.array(self.robot_poses, dtype=np.float64)
        camera_observations_np = np.array(self.camera_observations, dtype=np.float64)

        # 确保收集到的数据点足够
        # cv2.estimateAffine3D 需要至少 4 组点对
        if len(self.robot_poses) < 4 or len(self.camera_observations) < 4:
            print("错误: 手眼标定需要至少4组对应点。")
            return False

        try:
            # 从 self.robot_poses (4x4齐次矩阵) 中提取末端执行器的 XYZ 位置
            # 这是机械臂基座坐标系C下的点。`pose[:3, 3]` 提取的是 [x, y, z]
            robot_points_C = np.array([pose[:3, 3] for pose in self.robot_poses], dtype=np.float64)

            # self.camera_observations 应该已经存储了世界坐标系B下的3D点 [world_x, world_y, 0.0]
            # 因此，camera_observations_np 已经是 N x 3 的形式
            camera_points_B = camera_observations_np

            # 调试信息：打印数组形状和数据类型
            print(f"robot_points_C shape: {robot_points_C.shape}, dtype: {robot_points_C.dtype}")
            print(f"camera_points_B shape: {camera_points_B.shape}, dtype: {camera_points_B.dtype}")

            # 打印实际内容的前几行（如果存在）
            if robot_points_C.shape[0] > 0:
                print(f"DEBUG: First {{min(3, robot_points_C.shape[0])}} robot_points_C:\n{{robot_points_C[:min(3, robot_points_C.shape[0])]}}")
            if camera_points_B.shape[0] > 0:
                print(f"DEBUG: First {{min(3, camera_points_B.shape[0])}} camera_points_B:\n{{camera_points_B[:min(3, camera_points_B.shape[0])]}}")

            # 再次检查点集是否为空
            if robot_points_C.shape[0] == 0 or camera_points_B.shape[0] == 0:
                print("错误: 输入点集为空，无法执行手眼标定。")
                return False
            
            # 检查点数是否一致
            if robot_points_C.shape[0] != camera_points_B.shape[0]:
                print(f"错误: 机械臂点数 ({robot_points_C.shape[0]}) 与相机观测点数 ({camera_points_B.shape[0]}) 不一致。")
                return False

            # 检查点维度是否正确 (应该是 N x 3)
            if robot_points_C.shape[1] != 3 or camera_points_B.shape[1] != 3:
                print("错误: 输入点集维度不正确，应为 N x 3。")
                return False

            # 使用 estimateAffine3D 计算仿射变换矩阵
            # 它找到一个 3x4 的矩阵 [R|t]，使得 src_points * [R^T; t^T] = dst_points
            # 或者 dst_points = src_points * R + t (当 R 为 3x3，t 为 1x3 时)
            # cv2.estimateAffine3D(src, dst) 返回一个 3x4 的仿射变换矩阵 `out`，形式为 [R|t]
            # 其中 R 是 3x3 旋转矩阵，t 是 3x1 平移向量
            retval, self.hand_eye_matrix = cv2.estimateAffine3D(camera_points_B, robot_points_C)
            
            if retval:
                print("\n手眼标定完成！世界坐标系B到机械臂基座坐标系C的转换矩阵为 (3x4 仿射矩阵):")
                print(self.hand_eye_matrix)
                self.save_calibration_results()
                return True
            else:
                print("错误: 无法计算手眼标定仿射变换矩阵，请检查数据点是否足够或分布不合理。")
                return False

        except Exception as e:
            print(f"手眼标定计算过程中发生错误: {e}")
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