import numpy as np
import cv2
import os
import json
from datetime import datetime


class CameraCalibration:
    """
    基于棋盘格的相机内参标定类
    用于将图像坐标系A转换到世界坐标系B
    """
    
    def __init__(self, camera_index=0, chessboard_size=(9, 6), square_size=25.0):
        """
        初始化相机标定参数
        
        参数:
            camera_index (int): 摄像头索引，默认为0
            chessboard_size (tuple): 棋盘格内角点数量 (宽度, 高度)
            square_size (float): 棋盘格每个方格的实际尺寸，单位毫米
        """
        self.camera_index = camera_index
        self.chessboard_size = chessboard_size
        self.square_size = square_size
        
        # 标定数据存储
        self.object_points = []  # 3D世界坐标点
        self.image_points = []   # 2D图像坐标点
        self.captured_images = []  # 保存的图像
        
        # 标定结果
        self.camera_matrix = None
        self.distortion_coeffs = None
        self.calibration_error = None
        self.camera_matrix_loaded = False # 新增属性，指示相机矩阵是否已加载
        
        # 创建保存目录
        self.save_dir = "calibration_data"
        os.makedirs(self.save_dir, exist_ok=True)
        
        # 准备3D世界坐标点模板
        self._prepare_object_points()
        
        print(f"相机标定程序初始化完成")
        print(f"棋盘格规格: {chessboard_size[0]}x{chessboard_size[1]} 内角点")
        print(f"方格尺寸: {square_size}mm")
        print(f"数据保存目录: {self.save_dir}")
    
    def _prepare_object_points(self):
        """
        准备棋盘格的3D世界坐标点模板
        假设棋盘格放置在Z=0平面上
        """
        self.objp = np.zeros((self.chessboard_size[0] * self.chessboard_size[1], 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:self.chessboard_size[0], 0:self.chessboard_size[1]].T.reshape(-1, 2)
        self.objp *= self.square_size  # 转换为实际物理尺寸
    
    def capture_calibration_images(self, min_images=10):
        """
        实时捕获标定图像
        
        参数:
            min_images (int): 最少需要的标定图像数量
        """
        cap = cv2.VideoCapture(self.camera_index)
        
        if not cap.isOpened():
            print(f"错误: 无法打开摄像头 {self.camera_index}")
            return False
        
        print(f"\n=== 相机标定图像采集 ===")
        print(f"操作说明:")
        print(f"- 将棋盘格放置在不同位置和角度")
        print(f"- 确保棋盘格完全在画面内且清晰")
        print(f"- 按下 SPACE 键保存当前帧")
        print(f"- 按下 ESC 键退出采集")
        print(f"- 需要至少 {min_images} 张有效图像")
        print(f"- 当前已采集: 0 张\n")
        
        image_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("错误: 无法读取摄像头画面")
                break
            
            # 转换为灰度图像用于角点检测
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # 查找棋盘格角点
            ret_corners, corners = cv2.findChessboardCorners(
                gray, self.chessboard_size, None
            )
            
            # 显示画面
            display_frame = frame.copy()
            
            if ret_corners:
                # 精确化角点位置
                corners = cv2.cornerSubPix(
                    gray, corners, (11, 11), (-1, -1),
                    criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                )
                
                # 绘制角点
                cv2.drawChessboardCorners(display_frame, self.chessboard_size, corners, ret_corners)
                
                # 显示状态信息
                cv2.putText(display_frame, f"Found corners! Press SPACE to capture", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(display_frame, f"Images captured: {image_count}/{min_images}", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                # 显示未找到角点的提示
                cv2.putText(display_frame, "No chessboard detected", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(display_frame, f"Images captured: {image_count}/{min_images}", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            
            cv2.imshow('Camera Calibration - Press SPACE to capture, ESC to exit', display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == 27:  # ESC键退出
                break
            elif key == ord(' ') and ret_corners:  # 空格键保存图像
                # 保存图像数据
                self.object_points.append(self.objp)
                self.image_points.append(corners)
                self.captured_images.append(frame.copy())
                
                # 保存图像文件
                image_filename = os.path.join(self.save_dir, f"calibration_{image_count:03d}.jpg")
                cv2.imwrite(image_filename, frame)
                
                image_count += 1
                print(f"已保存第 {image_count} 张标定图像: {image_filename}")
                
                if image_count >= min_images:
                    print(f"\n已采集足够的标定图像 ({image_count} 张)")
                    print("按ESC退出采集，或继续添加更多图像")
        
        cap.release()
        cv2.destroyAllWindows()
        
        if image_count < min_images:
            print(f"警告: 只采集到 {image_count} 张图像，少于推荐的 {min_images} 张")
            return False
        
        print(f"图像采集完成，共采集 {image_count} 张有效图像")
        return True
    
    def calibrate_camera(self):
        """
        执行相机标定
        """
        if len(self.image_points) < 3:
            print("错误: 标定图像数量不足，至少需要3张图像")
            return False
        
        print("\n=== 开始相机标定 ===")
        print(f"使用 {len(self.image_points)} 张图像进行标定...")
        
        # 获取图像尺寸
        image_size = (self.captured_images[0].shape[1], self.captured_images[0].shape[0])
        
        # 执行相机标定
        ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
            self.object_points, self.image_points, image_size, None, None
        )
        
        if ret:
            self.camera_matrix = camera_matrix
            self.distortion_coeffs = dist_coeffs
            self.calibration_error = ret
            self.camera_matrix_loaded = True # 标定成功，设置为True
            
            print("相机标定成功!")
            print(f"重投影误差: {ret:.4f} 像素")
            print(f"\n相机内参矩阵:")
            print(camera_matrix)
            print(f"\n畸变系数:")
            print(dist_coeffs.flatten())
            
            # 计算每张图像的重投影误差
            total_error = 0
            for i in range(len(self.object_points)):
                projected_points, _ = cv2.projectPoints(
                    self.object_points[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs
                )
                error = cv2.norm(self.image_points[i], projected_points, cv2.NORM_L2) / len(projected_points)
                total_error += error
            
            mean_error = total_error / len(self.object_points)
            print(f"平均重投影误差: {mean_error:.4f} 像素")
            
            # 保存标定结果
            self.save_calibration_results()
            return True
        else:
            print("相机标定失败!")
            return False
    
    def save_calibration_results(self):
        """
        保存标定结果到文件
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        calibration_file = os.path.join(self.save_dir, f"camera_calibration_{timestamp}.json")
        
        calibration_data = {
            "timestamp": timestamp,
            "chessboard_size": self.chessboard_size,
            "square_size": self.square_size,
            "num_images": len(self.image_points),
            "calibration_error": float(self.calibration_error),
            "camera_matrix": self.camera_matrix.tolist(),
            "distortion_coefficients": self.distortion_coeffs.flatten().tolist(),
            "image_size": [self.captured_images[0].shape[1], self.captured_images[0].shape[0]]
        }
        
        with open(calibration_file, 'w', encoding='utf-8') as f:
            json.dump(calibration_data, f, indent=4, ensure_ascii=False)
        
        print(f"标定结果已保存到: {calibration_file}")
        
        # 同时保存为numpy格式，方便程序读取
        np_file = os.path.join(self.save_dir, f"camera_params_{timestamp}.npz")
        np.savez(np_file,
                camera_matrix=self.camera_matrix,
                distortion_coeffs=self.distortion_coeffs,
                calibration_error=self.calibration_error)
        print(f"标定参数已保存到: {np_file}")
    
    def load_calibration_results(self, calibration_file):
        """
        加载相机标定结果
        
        参数:
            calibration_file (str): 标定结果文件的路径
        """
        if not os.path.exists(calibration_file):
            raise FileNotFoundError(f"标定文件未找到: {calibration_file}")

        try:
            with open(calibration_file, 'r') as f:
                data = json.load(f)
                self.camera_matrix = np.array(data['camera_matrix'])
                self.distortion_coeffs = np.array(data['distortion_coeffs']).reshape(-1, 1)
                self.calibration_error = data['reprojection_error']
                self.camera_matrix_loaded = True # 加载成功，设置为True
                print(f"成功加载相机标定参数: {calibration_file}")
                return True
        except Exception as e:
            print(f"加载相机标定结果时发生错误: {e}")
            self.camera_matrix_loaded = False # 加载失败，设置为False
            return False
    
    def undistort_image(self, image):
        """
        对图像进行畸变校正
        
        参数:
            image: 输入图像
        返回:
            校正后的图像
        """
        if self.camera_matrix is None or self.distortion_coeffs is None:
            print("错误: 请先进行相机标定或加载标定结果")
            return image
        
        return cv2.undistort(image, self.camera_matrix, self.distortion_coeffs)
    
    def pixel_to_world_coordinates(self, pixel_x, pixel_y, world_z=0):
        """
        将像素坐标转换为世界坐标
        注意: 这需要假设目标在已知的Z平面上
        
        参数:
            pixel_x, pixel_y: 像素坐标
            world_z: 世界坐标系中的Z值（默认为0，即假设目标在标定平面上）
        返回:
            (world_x, world_y): 世界坐标
        """
        if self.camera_matrix is None:
            print("错误: 请先进行相机标定")
            return None
        
        # 这是一个简化的转换，实际应用中可能需要更复杂的平面校正
        # 使用相机内参进行反投影
        fx = self.camera_matrix[0, 0]
        fy = self.camera_matrix[1, 1]
        cx = self.camera_matrix[0, 2]
        cy = self.camera_matrix[1, 2]
        
        # 假设相机到平面的距离已知，这里需要根据实际标定结果调整
        # 实际使用时，可能需要通过透视变换或其他方法来获得更准确的转换
        world_x = (pixel_x - cx) * world_z / fx
        world_y = (pixel_y - cy) * world_z / fy
        
        return world_x, world_y
    
    def test_calibration(self):
        """
        测试标定效果
        """
        if self.camera_matrix is None:
            print("错误: 请先进行相机标定")
            return
        
        cap = cv2.VideoCapture(self.camera_index)
        
        print("\n=== 标定效果测试 ===")
        print("按ESC退出测试")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # 显示原始图像和校正后图像
            undistorted = self.undistort_image(frame)
            
            # 并排显示
            display = np.hstack([frame, undistorted])
            cv2.putText(display, "Original", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(display, "Undistorted", (frame.shape[1] + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.imshow('Calibration Test - Press ESC to exit', display)
            
            if cv2.waitKey(1) & 0xFF == 27:  # ESC
                break
        
        cap.release()
        cv2.destroyAllWindows()


def main():
    """
    主函数 - 执行完整的相机标定流程
    """
    print("=== 相机内参标定程序 ===")
    print("此程序将帮助您完成相机内参标定，实现图像坐标系A到世界坐标系B的转换")
    
    # 创建标定对象
    # 请根据你的棋盘格实际规格修改参数
    calibrator = CameraCalibration(
        camera_index=0,           # 摄像头索引，根据实际情况修改
        chessboard_size=(9, 6),   # 棋盘格内角点数量，根据实际棋盘格修改
        square_size=25.0          # 方格尺寸(mm)，根据实际棋盘格修改
    )
    
    # 步骤1: 采集标定图像
    print("\n步骤1: 采集标定图像")
    if not calibrator.capture_calibration_images(min_images=15):
        print("图像采集失败或数量不足")
        return
    
    # 步骤2: 执行标定
    print("\n步骤2: 执行相机标定")
    if not calibrator.calibrate_camera():
        print("相机标定失败")
        return
    
    # 步骤3: 测试标定效果
    print("\n步骤3: 测试标定效果")
    input("按回车键开始测试标定效果...")
    calibrator.test_calibration()
    
    print("\n相机内参标定完成!")
    print("标定结果已保存到 calibration_data 目录")
    print("你现在可以使用这些参数进行图像畸变校正和坐标转换")


if __name__ == "__main__":
    main()