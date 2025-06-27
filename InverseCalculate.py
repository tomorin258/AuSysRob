import math
from sympy import pi, ln, sqrt, cos, sin, acos, atan2, N  # N 用于数值评估


class BraccioInverseKinematics:
    """
    用于 Tinkerkit Braccio 机械臂的逆向运动学解析类。
    根据给定的末端执行器三维坐标，计算出机械臂各关节的角度。
    """

    def __init__(self):
        # === 参数配置区域开始 ===

        # 连杆长度参数 (a_i)
        self.a1 = 0  # 基座到关节2的水平偏移 (通常指DH参数中的a)
        self.a2 = 120  # 关节2 (肩部) 到关节3 (肘部) 的连杆长度
        self.a3 = 120  # 关节3 (肘部) 到关节4 (腕部) 的连杆长度
        self.a4 = 80  # 关节4 (腕部) 到关节5 (扭转) 的连杆长度
        self.a5 = 0  # 关节5 (扭转) 到末端执行器（夹持器尖端）的连杆长度

        # 连杆偏移参数 (d_i)
        self.d1 = 70  # 基座 Z 轴上的偏移量
        self.d2 = 0
        self.d3 = 0
        self.d4 = 0
        self.d5 = 150  # 扭转关节到末端执行器 Z 轴上的偏移量

        # 固定末端执行器方向向量（根据 Inverse.ipynb 的 w4, w5, w6）。
        # 在原始的逆运动学代码中，这些值是固定的，表示夹持器总是指向负Z方向（例如向下）。
        # 对于更通用的逆运动学，这些参数通常会从目标姿态中得出，但此处遵循原代码逻辑。
        self.w4 = 0
        self.w5 = 0
        self.w6 = -1
        # === 参数配置区域结束 ===

    def r2d(self, rad):
        """
        将弧度转换为角度（度）。
        """
        return rad / pi * 180

    def calculate_joint_angles(self, target_x, target_y, target_z):
        """
        根据给定的末端执行器目标三维坐标 (x, y, z) 计算前五个关节的角度。
        机械臂的参照坐标系原点为机械臂底座的中心点。

        参数:
            target_x (float): 目标点的 X 坐标。
            target_y (float): 目标点的 Y 坐标。
            target_z (float): 目标点的 Z 坐标。

        返回:
            list: 包含五个关节角度的列表 [base, shoulder, elbow, wrist, twist] (度)。
                  如果无法到达目标点，则返回 None。
        """
        # 将输入坐标直接映射到 Inverse.ipynb 中的 w1, w2, w3。
        # 原始 notebook 中有 wx*10, wy*10，但为了直接使用传入的三维坐标，
        # 我们假设 target_x, target_y, target_z 即为逆运动学公式中的 w1, w2, w3。
        w1 = target_x
        w2 = target_y
        w3 = target_z

        # --- 计算 q1 (基座关节角度) ---
        # 对应 Inverse.ipynb 中的 calculate_q1()
        q1 = atan2(w2, w1)

        # --- 计算 q_234 和 q3 (肘部关节角度) ---
        # 对应 Inverse.ipynb 中的 calculate_q3(q1)
        try:
            # 辅助角度 q_234 的计算，这与末端执行器方向有关
            # 原始代码中的 atan2(y, x)，其中 y = -(w4*cos(q1) + w5*sin(q1))，x = -w6
            q_234_y_component = -(self.w4 * cos(q1) + self.w5 * sin(q1))
            q_234 = atan2(q_234_y_component, -self.w6)

            # 计算辅助变量 b1, b2, b，它们描述了目标点在特定平面内的投影距离
            b1 = w1 * cos(q1) + w2 * sin(q1) - self.a4 * cos(q_234) + self.d5 * sin(q_234)
            b2 = self.d1 - self.a4 * sin(q_234) - self.d5 * cos(q_234) - w3
            b_squared = b1 ** 2 + b2 ** 2
            b = sqrt(b_squared)

            # 计算 q3 (肘部关节角度)
            # 检查 acos 的参数范围 [-1, 1]，防止因浮点误差或不可达点导致数学域错误
            cos_q3_arg = (b_squared - self.a2 ** 2 - self.a3 ** 2) / (2 * self.a2 * self.a3)
            # 允许一个小的浮点误差容差
            if abs(N(cos_q3_arg)) > 1.0 + 1e-9:
                print(f"目标点 ({target_x}, {target_y}, {target_z}) 无法到达 (q3 计算参数超出范围)。")
                return None

            # 原始代码中 q3 是 -acos，这通常表示机械臂“肘部向下弯曲”的解。
            # 存在另一个“肘部向上”的解，即 +acos(...)，此处遵循原始代码。
            q3 = -acos(cos_q3_arg)
        except Exception as e:
            print(f"计算 q3 失败，可能目标点不可达或参数异常: {e}")
            return None

        # --- 计算 q2 (肩部关节角度) ---
        # 对应 Inverse.ipynb 中的 calculate_q2(q3, b1, b2)
        try:
            numerator_q2 = (self.a2 + self.a3 * cos(q3)) * b2 - self.a3 * b1 * sin(q3)
            denominator_q2 = (self.a2 + self.a3 * cos(q3)) * b1 + self.a3 * b2 * sin(q3)
            q2 = atan2(numerator_q2, denominator_q2)
        except Exception as e:
            print(f"计算 q2 失败: {e}")
            return None

        # --- 计算 q4 (腕部关节角度) ---
        # 对应 Inverse.ipynb 中的 calculate_q4(q_234, q2, q3)
        q4 = q_234 - q2 - q3

        # --- 计算 q5 (扭转关节角度) ---
        # 对应 Inverse.ipynb 中的 calculate_q5()
        # 原始代码是 pi * ln(sqrt(w4**2 + w5**2 + w6**2))
        # 由于 w4=0, w5=0, w6=-1，所以 sqrt(w4**2 + w5**2 + w6**2) = sqrt(0^2 + 0^2 + (-1)^2) = sqrt(1) = 1
        # ln(1) = 0，因此在这种特定配置下，q5 总是 0。
        # 这表明原始的逆运动学简化了扭转关节的计算。
        try:
            magnitude_w_orientation = sqrt(self.w4 ** 2 + self.w5 ** 2 + self.w6 ** 2)
            q5 = pi * ln(magnitude_w_orientation)
        except Exception as e:
            print(f"计算 q5 失败: {e}")
            return None

        # --- 应用原始代码中的角度偏移 ---
        # 这些偏移可能是为了与机械臂的物理零位或正运动学约定相匹配。
        # 这是为了使计算出的角度与 Braccio 机械臂的实际控制角度相对应。
        q2_adjusted = q2 + pi / 2
        q3_adjusted = q3 + 3 * pi / 2

        # --- 将所有角度从弧度转换为度 ---
        # 使用 N() 对 sympy 表达式进行数值评估，以获得浮点数结果。
        q1_deg = self.r2d(N(q1))
        q2_deg = self.r2d(N(q2_adjusted))
        q3_deg = self.r2d(N(q3_adjusted))
        q4_deg = self.r2d(N(q4))
        q5_deg = self.r2d(N(q5))

        # --- 角度归一化 ---
        # 将角度归一化到 0-360 度范围内，方便后续处理。
        # 根据 Braccio 机械臂的实际物理限制（通常是 0-180 度），可能还需要进行额外的裁剪或检查。
        # 由于你要求“按照原理代码的数学方法解析”，这里只进行基本的归一化。
        q1_deg = q1_deg % 360
        q2_deg = q2_deg % 360
        q3_deg = q3_deg % 360
        q4_deg = q4_deg % 360
        q5_deg = q5_deg % 360

        return [q1_deg, q2_deg, q3_deg, q4_deg, q5_deg]


# 示例用法
if __name__ == "__main__":
    # 创建逆运动学解析器实例
    inverse_kinematics_solver = BraccioInverseKinematics()

    # 定义一个目标三维坐标点 (X, Y, Z)，单位为毫米 (mm)。
    # 你可以修改这些值来测试机械臂在不同位置的可达性。
    # 示例值 (你可以根据实际需求修改)：
    # 请注意：机械臂的可达工作空间有限，不合理的目标点将导致计算失败并返回 None。
    target_x_coord = 20.0
    target_y_coord = 0.0
    target_z_coord = 15.0

    print(f"计算目标点 ({target_x_coord}, {target_y_coord}, {target_z_coord}) 的关节角度...")
    joint_angles_deg = inverse_kinematics_solver.calculate_joint_angles(
        target_x_coord, target_y_coord, target_z_coord
    )

    if joint_angles_deg:
        print("\n计算出的关节角度 (度):")
        print(f"  基座 (Base):     {joint_angles_deg[0]:.2f}°")
        print(f"  肩部 (Shoulder): {joint_angles_deg[1]:.2f}°")
        print(f"  肘部 (Elbow):    {joint_angles_deg[2]:.2f}°")
        print(f"  腕部 (Wrist):    {joint_angles_deg[3]:.2f}°")
        print(f"  扭转 (Twist):    {joint_angles_deg[4]:.2f}°")

        # 这些角度 (转换为整数) 可以直接发送给前向控制器进行运动控制
        print("\n转换为整数后可发送给控制器 (示例):")
        print([int(angle) for angle in joint_angles_deg])
    else:
        print("未能找到可行的关节角度，目标点可能不可达或计算过程中出现问题。")

    # 你可以尝试其他目标点，包括可能不可达的点，例如：
    # print("\n尝试一个可能不可达的目标点 (例如，距离过远):")
    # joint_angles_unreachable = inverse_kinematics_solver.calculate_joint_angles(500, 0, 500)
    # if not joint_angles_unreachable:
    #     print("测试成功：不可达点被识别。")