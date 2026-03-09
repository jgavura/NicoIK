import socket, json, time
from datetime import datetime

HOST = "127.0.0.1"
PORT = 5005

# Attempt to import Motion, but handle failure gracefully if not simulating
try:
    from nicomotion.Motion import Motion
    NICOMOTION_AVAILABLE = True
except ImportError:
    Motion = None # Define Motion as None if import fails
    NICOMOTION_AVAILABLE = False
    print("Warning: nicomotion library not found. Hardware control disabled.")


# from grasper import Grasper


class Robot:
    INIT_POS = {  # right hand up, left down
        'head_z': 0.0, 'head_y': 0.0, 'r_shoulder_z': -24.0, 'r_shoulder_y': 13.0,
        'r_arm_x': 0.0, 'r_elbow_y': 104.0, 'r_wrist_z': -4.0, 'r_wrist_x': -55.0,
        'r_thumb_z': -62.0, 'r_thumb_x': -180.0, 'r_indexfinger_x': -170.0, 'r_middlefingers_x': -180.0,
        'l_shoulder_z': -24.0, 'l_shoulder_y': 13.0, 'l_arm_x': 0.0, 'l_elbow_y': 104.0,
        'l_wrist_z': -4.0, 'l_wrist_x': -55.0, 'l_thumb_z': -62.0, 'l_thumb_x': -180.0,
        'l_indexfinger_x': -170.0, 'l_middlefingers_x': -180.0
    }

    LIMITS = {
        'r_shoulder_z': {'min': -25.0, 'max': 80.0},
        'r_shoulder_y': {'min': -30.0, 'max': 180.0},
        'r_arm_x': {'min': -5.0, 'max': 70.0},
        'r_elbow_y': {'min': 50.0, 'max': 181},
        'r_wrist_z': {'min': -180.0, 'max': 180.0},
        'r_wrist_x': {'min': -180.0, 'max': 180.0},
        'r_thumb_z': {'min': -180.0, 'max': 180.0},
        'r_thumb_x': {'min': -180.0, 'max': 180.0},
        'r_indexfinger_x': {'min': -180.0, 'max': 180.0},
        'r_middlefingers_x': {'min': -180.0, 'max': 180.0}
    }

    def __init__(self, motor_config='./nico_humanoid_upper_rh7d_ukba.json'):
        self.head = ['head_z', 'head_y']    
        self.right_arm = ['r_shoulder_z', 'r_shoulder_y', 'r_arm_x', 'r_elbow_y', 'r_wrist_z', 'r_wrist_x']
        self.right_gripper = ['r_thumb_z', 'r_thumb_x', 'r_indexfinger_x', 'r_middlefingers_x']
        self.left_arm = ['l_shoulder_z', 'l_shoulder_y', 'l_arm_x', 'l_elbow_y', 'l_wrist_z', 'l_wrist_x']
        self.left_gripper = ['l_thumb_z', 'l_thumb_x', 'l_indexfinger_x', 'l_middlefingers_x']
        self.robot = None # For nicomotion hardware interface
        self.is_robot_connected = False
        self.speed = 0.03

        if NICOMOTION_AVAILABLE and Motion is not None:
            try:
                self.robot = Motion(motorConfig=motor_config)
                self.is_robot_connected = True
                print(f"Robot hardware initialized using config: {motor_config}")
            except Exception as e:
                print(f"Could not initialize robot hardware: {e}")
                print("Proceeding without hardware connection.")
                self.is_robot_connected = False
        elif not NICOMOTION_AVAILABLE:
             print("Hardware connection requested, but nicomotion library is not available.")
             self.is_robot_connected = False
        else:
             # Not attempting to connect robot
             self.is_robot_connected = False
        
        if self.is_robot_connected:
            self.init_position_full()
    
    def init_position_full(self):
        # print("Initializing robot to full initial position.")
        try:
            for joint_name, angle in self.INIT_POS.items():
                self.robot.setAngle(joint_name, angle, self.speed)
        except Exception as e:
            print(f"Error setting hardware pose: {e}")
            return False
    
    def clamp_angle(self, joint_name, angle):
        limits = self.LIMITS[joint_name]
        clamped_angle = max(limits['min'], min(limits['max'], angle))

        return clamped_angle

    def move_right_hand(self, angles):
        try:
            for i, angle in enumerate(angles):
                joint_name = (self.right_arm + self.right_gripper)[i]
                clamped_angle = self.clamp_angle(joint_name, angle)

                if joint_name == 'r_wrist_z':
                    clamped_angle *= 2
                elif joint_name == 'r_wrist_x':
                    clamped_angle *= 4

                self.robot.setAngle(joint_name, clamped_angle, self.speed)

            now = datetime.now()
            # time = now.strftime("%H:%M:%S") + f":{now.microsecond // 10000:02d}"        # stotiny
            time = now.strftime("%H:%M:%S") + f":{now.microsecond // 1000:03d}"         # milisekundy
            print(f'Robot started moving at: {time}')
        except Exception as e:
            print(f"Error setting hardware pose: {e}")
            return False


def handle_message(msg: str):
    angles = []
    if msg:
        angles = msg.split(' ')
        print(f'Received message with id: {angles[0]}')
    robot.move_right_hand([float(angle) for angle in angles[1:]])

    return {"ok": True, "echo": msg}


def connect():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((HOST, PORT))
        s.listen(1)
        print(f"Listening on {HOST}:{PORT}")

        conn, addr = s.accept()
        print("Connected:", addr)

        buf = b""
        while True:
            data = conn.recv(4096)
            if not data:
                print("Disconnected")
                break
            buf += data

            # print(f'Data :{data}')
            # print(f'Buf :{buf}')

            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                msg = line.decode("utf-8", errors="ignore").strip()
                if not msg:
                    continue

                # print("UE ->", msg)
                resp = handle_message(msg)
                conn.sendall((json.dumps(resp) + "\n").encode("utf-8"))


robot = Robot()

while True:
    connect()
