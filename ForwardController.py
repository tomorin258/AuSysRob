import serial
import time
from sympy import *


class BraccioRobot:
    """
    # Tinkerkit Braccio 机械臂的正向运动学

    ## 定义基本的 DH 变换矩阵
    定义在DH参数中使用的变量
    """

    def __init__(self, com_port='COM6', baud_rate=115200, timeout=5):
        # 定义DH变换矩阵中使用的符号变量
        self.d, self.a, self.alpha, self.delta = symbols('d,a,alpha,delta')

        # 定义旋转矩阵和变换矩阵
        self.Rx = Matrix([[1, 0, 0, 0],
                          [0, cos(self.alpha), -sin(self.alpha), 0],
                          [0, sin(self.alpha), cos(self.alpha), 0],
                          [0, 0, 0, 1]])
        self.Rz = Matrix([[cos(self.delta), -sin(self.delta), 0, 0],
                          [sin(self.delta), cos(self.delta), 0, 0],
                          [0, 0, 1, 0],
                          [0, 0, 0, 1]])
        self.Tz = Matrix([[1, 0, 0, 0],
                          [0, 1, 0, 0],
                          [0, 0, 1, self.d],
                          [0, 0, 0, 1]])
        self.Tx = Matrix([[1, 0, 0, self.a],
                          [0, 1, 0, 0],
                          [0, 0, 1, 0],
                          [0, 0, 0, 1]])
        # 完整的DH变换矩阵模板
        self.TM = self.Tz * self.Rz * self.Tx * self.Rx

        # === 参数配置区域开始 ===
        # 定义每个关节的DH参数和关节角度偏移
        # 顺序：基座、肩部、肘部、腕部、扭转 (对应 Braccio 机械臂的关节顺序)
        # 每个字典包含 'd', 'a', 'alpha' (弧度), 'delta_offset' (弧度)
        # delta_offset 是添加到 d2r(deg) 后的弧度值，用于补偿原点位置的偏移

        self.dh_parameters = [
            # 关节 0: 基座 (对应 Tv1n0)
            {'d': 70, 'a': 0, 'alpha': pi / 2, 'delta_offset': pi},
            # 关节 1: 肩部 (对应 Tv2n1)
            {'d': 0, 'a': 120, 'alpha': 0, 'delta_offset': 0},  # 原始: d2r(delta2) + pi/2 - pi/2 = d2r(delta2)
            # 关节 2: 肘部 (对应 Tv3n2)
            {'d': 0, 'a': 120, 'alpha': 0, 'delta_offset': pi / 2 - pi},  # 原始: d2r(delta3) +pi/2 - pi
            # 关节 3: 腕部 (对应 Tv4n3)
            {'d': 0, 'a': 80, 'alpha': -pi / 2, 'delta_offset': -pi / 2},  # 原始: d2r(delta4) +0 - pi/2
            # 关节 4: 扭转 (对应 Tv5n4)
            {'d': 150, 'a': 0, 'alpha': 0, 'delta_offset': -pi / 2}  # 原始: d2r(delta5) - pi/2 + 0
        ]
        # === 参数配置区域结束 ===

        # 初始化串口通信
        try:
            self.s = serial.Serial(com_port, baud_rate, timeout=timeout)
            time.sleep(3)  # 等待串口初始化完成
            print(f"串口 {com_port} 连接成功。")
            print(f"DEBUG: Serial port {com_port} initialized with baud rate {baud_rate}, timeout {timeout}")
        except serial.SerialException as e:
            print(f"串口连接失败: {e}")
            print(f"DEBUG: Failed to connect to serial port {com_port}: {e}")
            self.s = None

    def close_serial(self):
        """
        关闭串口连接
        """
        if self.s and self.s.is_open:
            self.s.close()
            print("串口已关闭。")
            print("DEBUG: Serial port closed.")

    def d2r(self, deg):
        """
        将角度（度）转换为弧度
        """
        return deg / 180 * pi

    """
    ## 定义每个连杆的变换函数 (已整合为通用函数)
    """

    def _get_joint_transformation_matrix(self, joint_idx, angle_deg):
        """
        获取指定关节的齐次变换矩阵。
        这个函数整合了原 Tv5n4, Tv4n3, Tv3n2, Tv2n1, Tv1n0 的功能。

        参数：
            joint_idx (int): 关节索引 (0-4)。
                             0: 基座 (Tv1n0)
                             1: 肩部 (Tv2n1)
                             2: 肘部 (Tv3n2)
                             3: 腕部 (Tv4n3)
                             4: 扭转 (Tv5n4)
            angle_deg (int): 关节角度（度）。
        返回：
            sympy.Matrix: 关节的齐次变换矩阵。
        """
        if not (0 <= joint_idx < len(self.dh_parameters)):
            raise ValueError("无效的关节索引。关节索引应在 0 到 4 之间。")

        params = self.dh_parameters[joint_idx]
        d_val = params['d']
        a_val = params['a']
        alpha_val = params['alpha']
        # 计算带偏移量的关节角（弧度）
        delta_val = self.d2r(angle_deg) + params['delta_offset']

        # 将参数代入通用的DH变换矩阵模板
        return self.TM.subs({self.delta: delta_val, self.d: d_val,
                             self.alpha: alpha_val, self.a: a_val})

    """
    ## 使用正向变换计算关节角度的 TCP 位置函数
    """

    def calculate_forward_kinematics(self, base, shoulder, elbow, wrist, twist):
        """
        根据关节角度计算末端执行器（TCP）的齐次变换矩阵。
        这个函数整合了原 BraccioForward 的功能。

        参数：
            base (int): 基座角度
            shoulder (int): 肩部角度
            elbow (int): 肘部角度
            wrist (int): 腕部角度
            twist (int): 扭转角度
        返回：
            sympy.Matrix: 末端执行器的齐次变换矩阵。
        """
        # 按照机械臂连杆顺序进行矩阵乘法
        # 注意：这里是 T_n_to_n-1 的顺序，与原 notebook 中的 TvXnY 调用顺序一致
        T_base = self._get_joint_transformation_matrix(0, base)
        T_shoulder = self._get_joint_transformation_matrix(1, shoulder)
        T_elbow = self._get_joint_transformation_matrix(2, elbow)
        T_wrist = self._get_joint_transformation_matrix(3, wrist)
        T_twist = self._get_joint_transformation_matrix(4, twist)

        return T_base * T_shoulder * T_elbow * T_wrist * T_twist

    """
    ## 测试 Braccio 机械臂的正向变换
    定义一个函数，该函数计算夹持器的尖端位置并用给定角度驱动机器人
    输入关节角度值，机器人应移动到计算出的位置。
    """

    def move_to_angles(self, base, shoulder, elbow, wrist, twist, gripper_pos=50, move_time=50):
        """
        计算末端执行器位置并驱动机器人到指定关节角度。
        这个函数整合了原 TestBraccioForward 的功能。

        参数：
            base (int): 基座角度
            shoulder (int): 肩部角度
            elbow (int): 肘部角度
            wrist (int): 腕部角度
            twist (int): 扭转角度
            gripper_pos (int): 夹持器位置 (0-180)，默认为 50
            move_time (int): 移动时间，默认为 50
        """
        if not self.s or not self.s.is_open:
            print("串口未连接或已关闭，无法发送命令。")
            print("DEBUG: Serial port not open, cannot send command.")
            return

        # 计算末端执行器位置
        # 乘以末端执行器的局部坐标系原点 [0,0,0,1]T
        P0_matrix = self.calculate_forward_kinematics(base, shoulder, elbow, wrist, twist) * Matrix([0, 0, 0, 1])
        print("计算出的末端执行器位置:")
        print(pretty(N(P0_matrix)))

        # 构造控制命令字符串
        command = (f"P{int(base)},{int(shoulder)},{int(elbow)},"
                   f"{int(wrist)},{int(twist)},{int(gripper_pos)},{int(move_time)}\n")
        print(f"DEBUG: Sending command: '{command.strip()}'") # 打印发送的命令

        try:
            self.s.write(command.encode('ascii'))
            # 尝试读取响应
            response = self.s.readline().decode().strip()
            print(f"机器人响应: {response}")
            print(f"DEBUG: Received response: '{response}'")
        except serial.SerialException as e:
            print(f"发送命令失败: {e}")
            print(f"DEBUG: Serial communication error when sending command: {e}")
        except Exception as e:
            print(f"发送命令时发生错误: {e}")
            print(f"DEBUG: General error when sending command: {e}")

    def move_to_home(self):
        """
        将机器人移动到起始位置。
        原始笔记本中的 Home position 是 P0,90,180,90,0,50,50
        """
        print("正在将机器人移动到 Home 位置 (0, 90, 180, 90, 0)。")
        self.move_to_angles(0, 90, 180, 90, 0, 50, 50)


# 以下是如何使用这个类
if __name__ == "__main__":
    # 请替换 'COM6' 为你的机械臂连接的实际串口号，例如 '/dev/ttyUSB0' (Linux) 或 'COMx' (Windows)
    robot = BraccioRobot(com_port='COM8', baud_rate=115200, timeout=5)

    # 将机器人移动到起始位置
    robot.move_to_home()

    # 命令移动到特定位置
    robot.move_to_angles(100, 90, 180, 90, 0) # 原始 TestBraccioForward 测试的参数

    # 在程序结束时关闭串口，确保资源释放
    robot.close_serial()