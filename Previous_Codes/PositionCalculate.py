import cv2
import numpy as np
from ultralytics import YOLO
import pickle
import os

class PositionCalculator:
    """
    用于连接C270摄像头，进行YOLO目标检测，并包含相机内参标定与平面校准的方法。
    """

    def __init__(self, camera_id=0, model_path="yolo11n.pt", calibration_file="camera_calibration.pkl", homography_file="homography_matrix.pkl"):
        self.camera_id = camera_id
        self.model_path = model_path
        self.calibration_file = calibration_file
        self.homography_file = homography_file
        self.camera_matrix = None
        self.dist_coeffs = None
        self.homography_matrix = None
        self.model = None

        self._load_yolo_model()
        self._load_calibration_data()

    def _load_yolo_model(self):
        """加载YOLO模型。"""
        try:
            self.model = YOLO(self.model_path)
            print(f"YOLO模型已从 {self.model_path} 加载。")
        except Exception as e:
            print(f"加载YOLO模型失败: {e}")
            self.model = None

    def _load_calibration_data(self):
        """加载相机内参标定数据和单应性矩阵。"""
        # 加载相机内参
        if os.path.exists(self.calibration_file):
            try:
                with open(self.calibration_file, 'rb') as f:
                    data = pickle.load(f)
                    self.camera_matrix = data['camera_matrix']
                    self.dist_coeffs = data['dist_coeffs']
                    print(f"相机内参标定数据已从 {self.calibration_file} 加载。")
            except Exception as e:
                print(f"加载相机内参标定数据失败: {e}")
        else:
            print(f"未找到相机内参标定文件 {self.calibration_file}。请先运行 calibrate_camera()。")

        # 加载单应性矩阵
        if os.path.exists(self.homography_file):
            try:
                with open(self.homography_file, 'rb') as f:
                    self.homography_matrix = pickle.load(f)
                    print(f"单应性矩阵已从 {self.homography_file} 加载。")
            except Exception as e:
                print(f"加载单应性矩阵失败: {e}")
        else:
            print(f"未找到单应性矩阵文件 {self.homography_file}。请先运行 calculate_plane_homography()。")

    def calibrate_camera(self, checkerboard_size=(9, 6), square_size_mm=25, num_images=15):
        """
        进行相机内参标定。
        参数:
            checkerboard_size (tuple): 棋盘格的内角点数量 (cols, rows)。例如 (9, 6) 表示横向有9个内角点，纵向有6个内角点。
            square_size_mm (float): 棋盘格每个方块的物理边长，单位毫米。
            num_images (int): 需要捕获用于标定的图片数量。
        """
        objp = np.zeros((checkerboard_size[0] * checkerboard_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:checkerboard_size[0], 0:checkerboard_size[1]].T.reshape(-1, 2) * square_size_mm

        objpoints = [] # 3d points in real world space
        imgpoints = [] # 2d points in image plane

        cap = cv2.VideoCapture(self.camera_id)
        if not cap.isOpened():
            print(f"错误: 无法打开摄像头 {self.camera_id}。")
            return

        print(f"\n=== 开始相机内参标定 ===\n请将棋盘格在摄像头视野中从不同角度展示，并按 'c' 捕获图片。")
        print(f"您需要捕获 {num_images} 张图片。按 'q' 退出标定过程。")
        captured_count = 0

        while captured_count < num_images:
            ret, frame = cap.read()
            if not ret:
                print("未能获取摄像头画面。")
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, checkerboard_size, None)

            if ret:
                # 绘制找到的角点
                cv2.drawChessboardCorners(frame, checkerboard_size, corners, ret)

            cv2.putText(frame, f"Captured: {captured_count}/{num_images}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow('相机内参标定', frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('c') and ret:
                objpoints.append(objp)
                imgpoints.append(corners)
                captured_count += 1
                print(f"已捕获 {captured_count}/{num_images} 张图片。")
                # 短暂延迟，避免连续捕获相同帧
                cv2.waitKey(500)

            elif key == ord('q'):
                print("用户中断标定过程。")
                break

        cap.release()
        cv2.destroyAllWindows()

        if len(objpoints) < num_images:
            print(f"警告: 捕获的图片数量不足 ({len(objpoints)}/{num_images})，无法进行标定。请重新尝试。")
            return

        print("\n正在计算相机内参...")
        try:
            ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
                objpoints, imgpoints, gray.shape[::-1], None, None
            )

            if ret:
                self.camera_matrix = camera_matrix
                self.dist_coeffs = dist_coeffs
                calibration_data = {'camera_matrix': camera_matrix, 'dist_coeffs': dist_coeffs}
                with open(self.calibration_file, 'wb') as f:
                    pickle.dump(calibration_data, f)
                print(f"相机内参标定完成，数据已保存到 {self.calibration_file}")
                print("相机矩阵:\n", self.camera_matrix)
                print("畸变系数:\n", self.dist_coeffs)
            else:
                print("相机内参标定失败。")
        except Exception as e:
            print(f"相机内参标定过程中发生错误: {e}")

    def calculate_plane_homography(self, checkerboard_size=(9, 6), square_size_mm=25):
        """
        计算从图像坐标系到世界坐标系（工作平面）的单应性矩阵（坐标系A -> 坐标系B）。
        此步骤假设相机已完成内参标定，并且棋盘格平放于工作平面上。
        参数:
            checkerboard_size (tuple): 棋盘格的内角点数量 (cols, rows)。
            square_size_mm (float): 棋盘格每个方块的物理边长，单位毫米。
        """
        if self.camera_matrix is None or self.dist_coeffs is None:
            print("请先进行相机内参标定 (calibrate_camera())，或确保已加载标定文件。")
            return

        # 定义棋盘格在世界坐标系中的3D点（Z=0，因为是平面）
        objp = np.zeros((checkerboard_size[0] * checkerboard_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:checkerboard_size[0], 0:checkerboard_size[1]].T.reshape(-1, 2) * square_size_mm
        world_points_2d = objp[:, :2] # 我们只需要2D世界坐标用于单应性矩阵

        cap = cv2.VideoCapture(self.camera_id)
        if not cap.isOpened():
            print(f"错误: 无法打开摄像头 {self.camera_id}。")
            return

        print("\n=== 开始计算平面单应性矩阵 ===\n请将棋盘格平放于工作平面上，并按 'c' 捕获图片。按 'q' 退出。")
        homography_calculated = False

        while not homography_calculated:
            ret, frame = cap.read()
            if not ret:
                print("未能获取摄像头画面。")
                break

            # 1. 图像去畸变 (重要: 在寻找角点之前进行)
            h, w = frame.shape[:2]
            new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(self.camera_matrix, self.dist_coeffs, (w, h), 1, (w, h))
            undistorted_frame = cv2.undistort(frame, self.camera_matrix, self.dist_coeffs, None, new_camera_matrix)

            gray = cv2.cvtColor(undistorted_frame, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, checkerboard_size, None)

            if ret:
                cv2.drawChessboardCorners(undistorted_frame, checkerboard_size, corners, ret)

            cv2.imshow('平面单应性矩阵计算', undistorted_frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('c') and ret:
                # 2. 精炼角点检测结果
                corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                                                    (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))

                # 3. 计算单应性矩阵
                H, _ = cv2.findHomography(corners_refined, world_points_2d, cv2.RANSAC, 5.0)
                if H is not None:
                    self.homography_matrix = H
                    with open(self.homography_file, 'wb') as f:
                        pickle.dump(H, f)
                    print(f"平面单应性矩阵计算完成，数据已保存到 {self.homography_file}")
                    print("单应性矩阵:\n", self.homography_matrix)
                    homography_calculated = True
                else:
                    print("无法计算单应性矩阵。请确保棋盘格清晰可见，并占据大部分画面。")
                cv2.waitKey(500) # 短暂延迟
            elif key == ord('q'):
                print("用户中断计算过程。")
                break

        cap.release()
        cv2.destroyAllWindows()

    def detect_and_calculate_position(self):
        """
        连接摄像头，进行 YOLO 目标检测，并将像素坐标转换为物理世界坐标（坐标系B）。
        """
        if self.camera_matrix is None or self.dist_coeffs is None:
            print("相机内参未加载。请先运行 calibrate_camera() 或确保已加载标定文件。")
            return
        if self.homography_matrix is None:
            print("单应性矩阵未加载。请先运行 calculate_plane_homography() 或确保已加载单应性矩阵文件。")
            return
        if self.model is None:
            print("YOLO 模型未加载。")
            return

        cap = cv2.VideoCapture(self.camera_id)
        if not cap.isOpened():
            print(f"错误: 无法打开摄像头 {self.camera_id}。")
            return

        print("\n=== 开始目标检测和坐标计算 ===\n按 'q' 退出。")

        while True:
            ret, frame = cap.read()
            if not ret:
                print("未能获取摄像头画面。")
                break

            # 1. 图像去畸变
            h, w = frame.shape[:2]
            # 根据新的相机矩阵调整画面，可以去除边缘的黑色区域
            new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(self.camera_matrix, self.dist_coeffs, (w, h), 1, (w, h))
            undistorted_frame = cv2.undistort(frame, self.camera_matrix, self.dist_coeffs, None, new_camera_matrix)

            # 如果需要，可以裁剪去畸变后的图像以去除黑色边缘
            # x, y, w_roi, h_roi = roi
            # undistorted_frame = undistorted_frame[y:y+h_roi, x:x+w_roi]

            # 2. YOLO 目标检测
            results = self.model(undistorted_frame, verbose=False) # verbose=False 抑制YOLO的日志输出

            # 3. 处理检测结果并转换坐标
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    conf = box.conf[0].cpu().numpy()
                    cls = int(box.cls[0].cpu().numpy())
                    print(f"[PositionCalculate] 检测到类别: {cls}，置信度: {conf:.2f}")
                    if cls != 9 or conf < 0.25:
                        continue
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    center_x_pixel = (x1 + x2) / 2
                    center_y_pixel = (y1 + y2) / 2
                    pixel_coords = np.array([[[center_x_pixel, center_y_pixel]]], dtype=np.float32)
                    world_coords_2d = cv2.perspectiveTransform(pixel_coords, self.homography_matrix)[0][0]
                    world_x_mm = world_coords_2d[0]
                    world_y_mm = world_coords_2d[1]
                    label = f"Obj: ({world_x_mm:.1f}mm, {world_y_mm:.1f}mm)"
                    cv2.rectangle(undistorted_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    cv2.putText(undistorted_frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    print(f"检测到物体中心点 (像素): ({center_x_pixel:.1f}, {center_y_pixel:.1f}) -> 世界坐标系B (mm): ({world_x_mm:.1f}, {world_y_mm:.1f})")

            cv2.imshow('物体检测与位置计算', undistorted_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    calculator = PositionCalculator()

    # --- 校准步骤说明 ---
    # 1. **相机内参标定**：只运行一次。需要您准备棋盘格，并在摄像头前从不同角度展示，捕获足够数量的图片。
    #    运行这行代码来启动标定过程:
    # calculator.calibrate_camera(checkerboard_size=(9, 6), square_size_mm=30, num_images=15)

    # 2. **平面校准（计算单应性矩阵）**：在相机内参标定完成后只运行一次。
    #    将棋盘格平放于您希望机械臂抓取物体的工作平面上。
    #    运行这行代码来启动校准过程:
    calculator.calculate_plane_homography(checkerboard_size=(9, 6), square_size_mm=30)

    # 3. **目标检测与坐标计算**：完成上述两个校准步骤并生成 'camera_calibration.pkl' 和 'homography_matrix.pkl' 文件后，
    #    您可以多次运行此方法来实时检测并计算物体在工作平面上的物理坐标。
    # calculator.detect_and_calculate_position()

    print("\n程序结束。") 

    '''请按照以下步骤使用此文件：
        相机内参标定（一次性操作）：
        您需要准备一个棋盘格作为标定物。
        打开 PositionCalculate.py 文件，找到 if __name__ == "__main__": 部分。
        取消注释并运行 calculator.calibrate_camera() 这一行代码。
        运行脚本后，摄像头会打开，您需要将棋盘格在摄像头视野中从不同角度展示，并根据屏幕提示按 'c' 键捕获足够的图片（默认15张）。
        完成后，脚本会将相机内参（相机矩阵和畸变系数）保存到 camera_calibration.pkl 文件中。
        平面校准（一次性操作）：
        在完成相机内参标定后，取消注释并运行 calculator.calculate_plane_homography() 这一行代码。
        运行脚本后，将棋盘格平放在您希望机械臂抓取物体的工作平面上。
        根据屏幕提示按 'c' 键捕获一张图片，脚本将计算出从像素坐标到物理世界坐标的单应性矩阵，并保存到 homography_matrix.pkl 文件中。
        目标检测与坐标计算（日常使用）：
        完成上述两个标定步骤并生成 camera_calibration.pkl 和 homography_matrix.pkl 文件后，您可以取消注释并运行 calculator.detect_and_calculate_position() 这一行代码。
        脚本将实时连接摄像头，使用 YOLO 检测物体，并将检测到的乐高积木的像素坐标自动转换为您在 SystemPrompt.md 中定义的坐标系 B（物理世界坐标系，单位毫米）下的 (X, Y) 坐标。这些坐标会直接显示在视频流上和控制台输出中。
    '''