from robolink import *    # API to communicate with RoboDK
from robodk import *      # robodk robotics toolbox
import numpy as np
from PIL import Image


from camera_class import IntelCam
from constants import Constants


class Robot:
    def __init__(self):
        # initialize if we're using simulation and mock sensors
        self.C = Constants()

        # initialize other classes
        if not self.C.cam_sim:
            self.camera = IntelCam()

        # sim
        self.cam_pose = xyzrpw_2_pose(self.C.cam_pose)
        self.laser_pose = xyzrpw_2_pose(self.C.laser_pose)
        self.ee_pose = xyzrpw_2_pose(self.C.ee_pose)

        # frames
        self.base_frame = xyzrpw_2_pose(self.C.base_frame)
        self.point_frame = xyzrpw_2_pose(self.C.point_frame)
        
######################################################################################################################
        
    def initialize(self):
        # Any interaction with RoboDK must be done through RDK:
        self.RDK = Robolink()
        self.RDK.Cam2D_Close()

        # Select a robot (popup is displayed if more than one robot is available)
        self.robot = self.RDK.ItemUserPick('Select a robot', ITEM_TYPE_ROBOT)
        if not self.robot.Valid():
            raise Exception('No robot selected or available')
        # Important: by default, the run mode is RUNMODE_SIMULATE
        # If the program is generated offline manually the runmode will be RUNMODE_MAKE_ROBOTPROG,
        # Therefore, we should not run the program on the robot
        if self.RDK.RunMode() != RUNMODE_SIMULATE:
            self.C.robot_sim = True

        if not self.C.robot_sim:
            # Connect to the robot using default IP
            success = self.robot.Connect()  # Try to connect once
            # success robot.ConnectSafe() # Try to connect multiple times
            status, status_msg = self.robot.ConnectedState()
            if status != ROBOTCOM_READY:
                # Stop if the connection did not succeed
                raise Exception("Failed to connect: " + status_msg)

            # This will set to run the API programs on the robot and the simulator (online programming)
            self.RDK.setRunMode(RUNMODE_RUN_ROBOT)
            # Note: This is set automatically when we Connect() to the robot through the API
    
    def home(self):
        self.initialize()
        self.robot.setSpeed(50, 25)
        self.robot.MoveJ(self.C.joints_set, blocking=True)

    def main(self):
        self.home()

######################################################################################################################

if __name__ == "__main__":
    ur10 = Robot()
    ur10.main()

