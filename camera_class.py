import pyrealsense2 as rs
import numpy as np
import cv2
import time
import matplotlib.pyplot as plt
from PIL import Image


class IntelCam:
    def __init__(self):
        # Configure depth and color streams
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        
        # Get device product line for setting a supporting resolution
        pipeline_wrapper = rs.pipeline_wrapper(self.pipeline)
        pipeline_profile = self.config.resolve(pipeline_wrapper)
        device = pipeline_profile.get_device()
        device_product_line = str(device.get_info(rs.camera_info.product_line))

        found_rgb = False
        for s in device.sensors:
            if s.get_info(rs.camera_info.name) == 'RGB Camera':
                found_rgb = True
                break
        if not found_rgb:
            print("The demo requires Depth camera with Color sensor")
            exit(0)


    def initialize(self):
        self.config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
        self.config.enable_stream(rs.stream.color, 1920, 1080, rs.format.bgr8, 30)

        # Start streaming
        profile = self.pipeline.start(self.config)
        # Set the exposure anytime during the operation
        rgb_sensor = self.pipeline.get_active_profile().get_device().query_sensors()[1]
        rgb_sensor.set_option(rs.option.exposure, 100.000)



    def rgb_intrinsics(self):
        # get camera intrinsics
        profile = self.pipeline.get_active_profile()

        rgb_profile = rs.video_stream_profile(profile.get_stream(rs.stream.color))
        rgb_intrinsics = rgb_profile.get_intrinsics()

        K = np.array([[rgb_intrinsics.fx, 0, rgb_intrinsics.ppx],
                    [0, rgb_intrinsics.fy, rgb_intrinsics.ppy],
                    [0, 0, 1]])
        return K


    def depth_intrinsics(self):
        depth_profile = rs.video_stream_profile(self.profile.get_stream(rs.stream.depth))
        depth_intrinsics = depth_profile.get_intrinsics()
        return depth_intrinsics
    

    def close(self):
        # Stop streaming
        self.pipeline.stop()

    def capture_image(self, img_name):
        # Create an align object
        # rs.align allows us to perform alignment of depth frames to others frames
        # The "align_to" is the stream type to which we plan to align depth frames.
        align_to = rs.stream.color
        align = rs.align(align_to)
        hole_filling = rs.hole_filling_filter(2)

        # Wait for a coherent pair of frames: depth and color
        frames = self.pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        depth_frame = hole_filling.process(depth_frame)

        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # Align the depth frame to color frame
        aligned_frames = align.process(frames)

        # Get aligned frames
        aligned_depth_frame = aligned_frames.get_depth_frame() 
        aligned_color_frame = aligned_frames.get_color_frame()

        aligned_depth_frame = hole_filling.process(aligned_depth_frame)

        self.aligned_depth_image = np.asanyarray(aligned_depth_frame.get_data())
        
        aligned_color_image = np.asanyarray(aligned_color_frame.get_data())
        
        # Apply colormap to depth image
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.255), cv2.COLORMAP_JET)
        aligned_depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(self.aligned_depth_image, alpha=0.255), cv2.COLORMAP_JET)

        
        img_path = f"images/rgb/{img_name}"
        cv2.imwrite(img_path, color_image)
        cv2.imwrite(f"images/depth/depth.jpeg", aligned_depth_colormap)

        return img_path
    
    def depth_val(self, u, v):
        # Calculated start and end coordinates for cropping
        start_x = 704
        start_y = 284

        # Extract the corresponding 512x512 section of the depth values
        
        cropped_depth_values = self.aligned_depth_image[start_y:start_y+512, start_x:start_x+512]
        # print(cropped_depth_values[v, u])
        return cropped_depth_values[v, u]


if __name__=="__main__":
    camera = IntelCam()
    camera.initialize()
    K = camera.rgb_intrinsics()
    print(K)
    img_path = camera.capture_image("test.jpeg")
    img = Image.open(img_path)
    px = K[0][2]
    py = K[1][2]
    left, top, right, bottom = px-256, py-256, px+256, py+256

    # Perform the crop
    cropped_img = img.crop((left, top, right, bottom))
    cropped_img.save(f"images/test_cropped.jpeg")
    
    camera.depth_val(413, 308)
    camera.close()
    