import numpy as np

class Constants:
    def __init__(self):
        self.robot_sim = True
        self.cam_sim = True
        self.just_pic = True

        # joints
        self.joints_home = [0, -90, -90, 0, 90, 0]
        # self.joints_set = [29.20, -109.38, -144.37, 73.76, 60.80, 0.00]
        self.joints_set = [21.52, -102.32, -146.78, 70.16, 70.48, -0.85]

        # poses
        self.cam_pose = [-32.5, 41.5, 111.0, -90, 0, 0]
        self.laser_pose = [-43.5, 40.95, 53.10, -90, 90, 0]
        self.ee_pose = [0, 230.975, 10.584, 0, -90, 90]

        # frames
        self.base_frame = [0, 0, 0, 0, 0, 0]
        self.point_frame = [750, 0, -72.5, 0, 0, 0]

        self.rgb_img_name = "rgb.jpeg"
