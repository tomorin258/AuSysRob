import numpy as np
import cv2
import pickle
import os
from Previous_Codes.PositionCalculate import PositionCalculator # 引入之前生成的 PositionCalculator

class HandEyeCalibrator:
    """
    用于执行手眼标定，将世界坐标系B（平面）对齐到机械臂基座坐标系C。
    并提供将检测到的方块坐标转换为机械臂可用三维坐标的功能。
    """

    def __init__(self, calibration_file="hand_eye_calibration.pkl", lego_height_mm=20):
        self.calibration_file = calibration_file
        self.lego_height_mm = lego_height_mm # 固定的乐高积木高度，单位毫米
        self.transform_matrix = None # 从B到C的齐次变换矩阵
        self._load_calibration_data()

    def _load_calibration_data(self):
        """加载手眼标定数据。"""
        if os.path.exists(self.calibration_file):
            try:
                with open(self.calibration_file, 'rb') as f:
                    self.transform_matrix = pickle.load(f)
                    print(f"手眼标定数据已从 {self.calibration_file} 加载。")
            except Exception as e:
                print(f"加载手眼标定数据失败: {e}")
        else:
            print(f"未找到手眼标定文件 {self.calibration_file}。请先运行 collect_and_calibrate()。")

    def collect_and_calibrate(self, position_calculator: PositionCalculator, num_points=5):
        """
        引导用户收集手眼标定数据并计算转换矩阵。
        参数:
            position_calculator (PositionCalculator): 用于获取坐标系B数据的PositionCalculator实例。
            num_points (int): 需要收集的标定点对数量（至少3个，建议5个或更多）。
        """
        if position_calculator.camera_matrix is None or position_calculator.dist_coeffs is None:
            print("错误: PositionCalculator 尚未完成相机内参标定。请先完成。")
            return
        if position_calculator.homography_matrix is None:
            print("错误: PositionCalculator 尚未完成平面校准。请先完成。")
            return

        print("\n=== 开始手眼标定数据收集 ===")
        print("您需要收集至少 3 组对应点 (世界坐标系B <-> 机械臂基座坐标系C)。")
        print("建议使用棋盘格上的特定角点作为标定点，或在工作平面上放置可识别的标记点。")
        print("请确保 PositionCalculator 已经正确初始化并能检测到目标点。")

        points_B = [] # 世界坐标系B下的点 (x, y)
        points_C = [] # 机械臂基座坐标系C下的点 (x, y, z)

        cap = cv2.VideoCapture(position_calculator.camera_id)
        if not cap.isOpened():
            print(f"错误: 无法打开摄像头 {position_calculator.camera_id}。")
            return

        for i in range(num_points):
            print(f"\n--- 收集第 {i+1}/{num_points} 组点 ---")
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("未能获取摄像头画面。")
                    break

                # 图像去畸变
                h, w = frame.shape[:2]
                new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(position_calculator.camera_matrix, position_calculator.dist_coeffs, (w, h), 1, (w, h))
                undistorted_frame = cv2.undistort(frame, position_calculator.camera_matrix, position_calculator.dist_coeffs, None, new_camera_matrix)

                # 红色色块检测（替换YOLO部分）
                hsv = cv2.cvtColor(undistorted_frame, cv2.COLOR_BGR2HSV)
                # 红色有两个区间
                lower_red1 = np.array([0, 100, 100])
                upper_red1 = np.array([10, 255, 255])
                lower_red2 = np.array([160, 100, 100])
                upper_red2 = np.array([179, 255, 255])
                mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
                mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
                mask = cv2.bitwise_or(mask1, mask2)
                # 形态学操作去噪
                kernel = np.ones((5, 5), np.uint8)
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
                mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel)
                # 找轮廓
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                current_world_B_coords = None
                if contours:
                    # 取最大轮廓
                    largest_contour = max(contours, key=cv2.contourArea)
                    if cv2.contourArea(largest_contour) > 100:  # 面积阈值可调整
                        M = cv2.moments(largest_contour)
                        if M["m00"] != 0:
                            cX = int(M["m10"] / M["m00"])
                            cY = int(M["m01"] / M["m00"])
                            pixel_coords = np.array([[[cX, cY]]], dtype=np.float32)
                            world_coords_2d = cv2.perspectiveTransform(pixel_coords, position_calculator.homography_matrix)[0][0]
                            current_world_B_coords = (world_coords_2d[0], world_coords_2d[1])
                            label = f"Red: ({current_world_B_coords[0]:.1f}mm, {current_world_B_coords[1]:.1f}mm)"
                            cv2.drawContours(undistorted_frame, [largest_contour], -1, (0, 255, 0), 2)
                            cv2.circle(undistorted_frame, (cX, cY), 5, (0, 255, 255), -1)
                            cv2.putText(undistorted_frame, label, (cX, cY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                cv2.putText(undistorted_frame, "Press 's' to Save Point, 'q' to Quit", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.imshow('手眼标定数据收集', undistorted_frame)
                key = cv2.waitKey(1) & 0xFF

                if key == ord('s'):
                    if current_world_B_coords is not None:
                        print(f"\n--- 当前检测到的世界坐标系B点: ({current_world_B_coords[0]:.1f}, {current_world_B_coords[1]:.1f}) mm ---")
                        while True:
                            try:
                                c_x = float(input("请输入此点在机械臂基座坐标系C下的 X 坐标 (mm): "))
                                c_y = float(input("请输入此点在机械臂基座坐标系C下的 Y 坐标 (mm): "))
                                c_z = float(input("请输入此点在机械臂基座坐标系C下的 Z 坐标 (mm): "))
                                points_B.append(np.array([current_world_B_coords[0], current_world_B_coords[1], 0.0])) # Z=0 for plane
                                points_C.append(np.array([c_x, c_y, c_z]))
                                print(f"点对已保存: B({current_world_B_coords[0]:.1f}, {current_world_B_coords[1]:.1f}) -> C({c_x:.1f}, {c_y:.1f}, {c_z:.1f})")
                                break
                            except ValueError:
                                print("输入无效，请重新输入数字。")
                        break # 退出内部循环，进入下一个点收集
                    else:
                        print("未检测到物体，无法保存点。请确保物体在视野中。")
                elif key == ord('q'):
                    print("用户中断标定数据收集。")
                    cap.release()
                    cv2.destroyAllWindows()
                    return

        cap.release()
        cv2.destroyAllWindows()

        if len(points_B) < num_points:
            print(f"警告: 收集的点对数量不足 ({len(points_B)}/{num_points})，无法进行手眼标定。")
            return

        print("\n正在计算手眼转换矩阵...")
        try:
            # 将列表转换为Numpy数组
            points_B_np = np.array(points_B, dtype=np.float32)
            points_C_np = np.array(points_C, dtype=np.float32)

            # 使用 cv2.estimateAffine3D 或 cv2.estimateAffine2D (如果简化) 来寻找变换
            # 由于我们将B坐标扩展到了3D（Z=0），这里寻找3D到3D的变换
            # 这里我们寻找一个仿射变换，如果更精确，可以考虑 rigid transformation (欧氏变换)
            # 或者使用 solvePnP 结合特征点
            # 对于点对点的最小二乘解，我们可以直接计算

            # 假设我们寻找一个R*X_B + t = X_C 的变换
            # 我们需要至少3个非共线的点来求解2D到2D的仿射变换
            # 对于3D点，我们至少需要3个非共线的点来求解刚性变换或仿射变换

            # 这里使用一个简化的方法：计算质心，然后对齐。
            # 更精确的通用解法是基于最小二乘的点集配准，例如Kabsch算法或SVD。
            # 但对于B->C，考虑到B是平面，C是三维，且Z固定，可以简化为2D到3D的映射。

            # 简化方案：只考虑X,Y的平移和缩放，Z固定
            # 这里需要一个更健壮的3D点集配准方法。opencv提供了solvePnP，但它需要模型点和图像点。
            # 对于点对点，我们通常用更通用的方法。

            # 使用 solvePnP 替代手动计算 (需要一个平面模型点集和对应的图像点集)
            # 考虑到我们已经将A->B转换了，B是平面世界坐标。
            # 现在需要B->C。假设B是XY平面，C是XYZ空间。
            # 我们可以将B的(x,y)视为相机坐标系下的(X,Y,0)点，然后寻找R,t。

            # 对于这种点对点的手眼标定，我们可以直接使用最小二乘方法。
            # 创建增广矩阵以便求解齐次变换
            A = np.hstack((points_B_np, np.ones((points_B_np.shape[0], 1)))) # [x_B, y_B, z_B, 1]
            B = points_C_np # [x_C, y_C, z_C]

            # 求解 AX = B，其中 X 是变换矩阵
            # 使用 np.linalg.lstsq 求解最小二乘问题
            # X = (A^T * A)^-1 * A^T * B
            # 如果点数量足够，且点分布合理，这个方法可以得到较好的仿射变换
            try:
                # 对于精确的刚体变换 (旋转+平移)，需要更多高级算法
                # 这里我们假设从 B 到 C 是一个线性变换 (仿射变换的特例: 刚体变换)
                # 我们可以使用 cv2.estimateAffine3D，它需要至少 4 个点对
                if len(points_B) >= 4:

                    # 构建用于 points_B_np 的增广矩阵
                    points_B_aug = np.hstack((points_B_np, np.ones((points_B_np.shape[0], 1)))) # N x 4
                    
                    # 使用最小二乘法求解变换矩阵
                    # 我们希望求解 points_B_aug @ M_transpose = points_C_np
                    # 其中 M_transpose 是 4x3 矩阵
                    M_transpose, residuals, rank, s = np.linalg.lstsq(points_B_aug, points_C_np, rcond=None)
                    
                    self.transform_matrix = M_transpose.T # M 是 3x4 矩阵
                    
                    with open(self.calibration_file, 'wb') as f:
                        pickle.dump(self.transform_matrix, f)
                    print(f"手眼标定完成，转换矩阵已保存到 {self.calibration_file}")
                    print("转换矩阵 (B->C):\n", self.transform_matrix)

                else:
                    print("手眼标定需要至少 4 组点对来计算3D仿射变换。")
                    return
            except np.linalg.LinAlgError as e:
                print(f"手眼标定失败: 无法计算仿射变换矩阵，线性代数错误: {e}")
                print("请检查数据点是否足够或分布不合理。")
                return
            except Exception as e:
                print(f"计算手眼变换矩阵时发生错误: {e}")
                return

        except Exception as e:
            print(f"计算手眼转换矩阵失败: {e}")
            return

    def get_block_coordinate_in_robot_base(self, position_calculator: PositionCalculator):
        """
        实时获取 YOLO 检测到的乐高积木在机械臂基座坐标系C下的三维坐标。
        参数:
            position_calculator (PositionCalculator): 用于获取坐标系B数据的PositionCalculator实例。
        返回:
            tuple: 乐高积木在机械臂基座坐标系C下的 (x, y, z) 坐标 (毫米)。
                   如果未检测到物体或未完成标定，则返回 None。
        """
        if self.transform_matrix is None:
            print("手眼标定未完成或未加载。请先运行 collect_and_calibrate() 或确保已加载标定文件。")
            return None

        if position_calculator.camera_matrix is None or position_calculator.dist_coeffs is None:
            print("相机内参未加载。请先运行 PositionCalculator.calibrate_camera() 或确保已加载标定文件。")
            return None
        if position_calculator.homography_matrix is None:
            print("单应性矩阵未加载。请先运行 PositionCalculator.calculate_plane_homography() 或确保已加载单应性矩阵文件。")
            return None
        if position_calculator.model is None:
            print("YOLO 模型未加载。")
            return None

        cap = cv2.VideoCapture(position_calculator.camera_id)
        if not cap.isOpened():
            print(f"错误: 无法打开摄像头 {position_calculator.camera_id}。")
            return None

        print("\n=== 实时获取方块坐标 (机械臂基座坐标系C) ===\n检测到积木后自动返回。")
        robot_base_coords = None
        while True:
            ret, frame = cap.read()
            if not ret:
                print("未能获取摄像头画面。")
                break

            # 1. 图像去畸变
            h, w = frame.shape[:2]
            new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(position_calculator.camera_matrix, position_calculator.dist_coeffs, (w, h), 1, (w, h))
            undistorted_frame = cv2.undistort(frame, position_calculator.camera_matrix, position_calculator.dist_coeffs, None, new_camera_matrix)

            # 2. YOLO 目标检测
            results = position_calculator.model(undistorted_frame, verbose=False)

            for r in results:
                boxes = r.boxes
                for box in boxes:
                    conf = box.conf[0].cpu().numpy()
                    cls = int(box.cls[0].cpu().numpy())
                    print(f"[HandEyeCalibration] 检测到类别: {cls}，置信度: {conf:.2f}")
                    if cls != 9 or conf < 0.15:
                        continue
                    # 以下为原有处理逻辑
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    center_x_pixel = (x1 + x2) / 2
                    center_y_pixel = (y1 + y2) / 2
                    pixel_coords = np.array([[[center_x_pixel, center_y_pixel]]], dtype=np.float32)
                    world_coords_2d_B = cv2.perspectiveTransform(pixel_coords, position_calculator.homography_matrix)[0][0]
                    point_B_3d = np.array([world_coords_2d_B[0], world_coords_2d_B[1], 0, 1], dtype=np.float32)
                    transformed_point_C = np.dot(self.transform_matrix, point_B_3d)
                    robot_base_x = transformed_point_C[0]
                    robot_base_y = transformed_point_C[1]
                    robot_base_z = self.lego_height_mm
                    robot_base_coords = (robot_base_x, robot_base_y, robot_base_z)
                    label = f"Robot Coords: ({robot_base_x:.1f}, {robot_base_y:.1f}, {robot_base_z:.1f}) mm"
                    cv2.rectangle(undistorted_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    cv2.putText(undistorted_frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    print(f"检测到物体中心点 (机械臂基座坐标系C): ({robot_base_x:.1f}, {robot_base_y:.1f}, {robot_base_z:.1f}) mm")
                    cv2.imshow('实时方块坐标 (机械臂基座坐标系C)', undistorted_frame)
                    cv2.waitKey(500)
                    cap.release()
                    cv2.destroyAllWindows()
                    return robot_base_coords

            cv2.imshow('实时方块坐标 (机械臂基座坐标系C)', undistorted_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        return None

    def transform_point_B_to_C(self, point_B):
        """
        输入：point_B = [x, y, z]
        输出：point_C = [x, y, z]
        """
        if self.transform_matrix is None:
            return None
        point_B_homo = np.array([point_B[0], point_B[1], point_B[2], 1], dtype=np.float32)
        point_C_homo = np.dot(self.transform_matrix, point_B_homo)
        return point_C_homo[:3].tolist()

if __name__ == "__main__":
    # 确保 PositionCalculator 已经正确初始化并完成了相机内参标定和平面校准
    # 如果没有，请在 PositionCalculate.py 中运行相关方法
    pos_calculator = PositionCalculator()

    # --- 手眼标定步骤说明 ---
    # 1. **数据收集与标定计算**：只运行一次。
    #    您需要手动移动机械臂，将末端执行器定位到工作平面上的已知点（例如棋盘格的角点）。
    #    在 PositionCalculate 能够检测到该点时，在弹出的窗口中按 's' 保存，然后输入机械臂末端在基座坐标系C下的实际XYZ坐标。
    #    重复此过程 num_points 次 (建议至少5次，并分布在工作空间中)。
    #    运行这行代码来启动数据收集和标定过程:
    calibrator = HandEyeCalibrator(lego_height_mm=20) # 乐高积木的高度，单位毫米
    calibrator.collect_and_calibrate(pos_calculator, num_points=4)

    # 2. **获取方块坐标 (日常使用)**：完成手眼标定并生成 'hand_eye_calibration.pkl' 文件后，
    #    您可以多次运行此方法来实时检测并计算乐高积木在机械臂基座坐标系C下的三维坐标。
    # calibrator = HandEyeCalibrator(lego_height_mm=20) # 请确保这里的乐高积木高度与您的实际测量值一致
    # final_block_coords = calibrator.get_block_coordinate_in_robot_base(pos_calculator)

    # if final_block_coords:
    #     print(f"\n最终检测到的乐高积木坐标 (机械臂基座坐标系C): {final_block_coords[0]:.2f}, {final_block_coords[1]:.2f}, {final_block_coords[2]:.2f} mm")
    # else:
    #     print("未能获取乐高积木坐标。")

    print("\n程序结束。")