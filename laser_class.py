import ctypes as ct
import time
from matplotlib import pyplot as plt
import pyllt as llt

class scanControl:
    def __init__(self):
        # Parametrize transmission
        self.scanner_type = ct.c_int(0)
        
        # Init profile buffer and timestamp info
        self.noProfileReceived = True
        self.exposure_time = 100
        self.idle_time = 3900
        self.timestamp = (ct.c_ubyte*16)()
        self.available_resolutions = (ct.c_uint*4)()
        self.available_interfaces = (ct.c_uint*6)()
        self.lost_profiles = ct.c_int()
        self.shutter_opened = ct.c_double(0.0)
        self.shutter_closed = ct.c_double(0.0)
        self.profile_count = ct.c_uint(0)
        
        # Null pointer if data not necessary
        self.null_ptr_short = ct.POINTER(ct.c_ushort)()
        self.null_ptr_int = ct.POINTER(ct.c_uint)()

    def initialize(self):
        
        # Create instance 
        self.hLLT = llt.create_llt_device(llt.TInterfaceType.INTF_TYPE_ETHERNET)
        
        # Get available interfaces
        self.ret = llt.get_device_interfaces_fast(self.hLLT, self.available_interfaces, len(self.available_interfaces))
        if self.ret < 1:
            raise ValueError("Error getting interfaces : " + str(self.ret))
        
        # Set IP address
        self.ret = llt.set_device_interface(self.hLLT, self.available_interfaces[0], 0)
        if self.ret < 1:
            raise ValueError("Error setting device interface: " + str(self.ret))
        
        # Connect
        self.ret = llt.connect(self.hLLT)
        if self.ret < 1:
            raise ConnectionError("Error connect: " + str(self.ret))
        
        # Get available resolutions
        self.ret = llt.get_resolutions(self.hLLT, self.available_resolutions, len(self.available_resolutions))
        if self.ret < 1:
            raise ValueError("Error getting resolutions : " + str(self.ret))
        
        # Set max. resolution
        self.resolution = self.available_resolutions[0]
        self.ret = llt.set_resolution(self.hLLT, self.resolution)
        if self.ret < 1:
            raise ValueError("Error getting resolutions : " + str(self.ret))
        
        # Declare measuring data arrays
        self.profile_buffer = (ct.c_ubyte*(self.resolution*64))()
        self.x = (ct.c_double * self.resolution)()
        self.z = (ct.c_double * self.resolution)()
        self.intensities = (ct.c_ushort * self.resolution)()

        # Scanner type
        self.ret = llt.get_llt_type(self.hLLT, ct.byref(self.scanner_type))
        if self.ret < 1:
            raise ValueError("Error scanner type: " + str(self.ret))

        # Set profile config
        self.ret = llt.set_profile_config(self.hLLT, llt.TProfileConfig.PROFILE)
        if self.ret < 1:
            raise ValueError("Error setting profile config: " + str(self.ret))

        # Set trigger
        self.ret = llt.set_feature(self.hLLT, llt.FEATURE_FUNCTION_TRIGGER, llt.TRIG_INTERNAL)
        if self.ret < 1:
            raise ValueError("Error setting trigger: " + str(self.ret))

        # Set exposure time
        self.ret = llt.set_feature(self.hLLT, llt.FEATURE_FUNCTION_EXPOSURE_TIME, self.exposure_time)
        if self.ret < 1:
            raise ValueError("Error setting exposure time: " + str(self.ret))

        # Set idle time
        self.ret = llt.set_feature(self.hLLT, llt.FEATURE_FUNCTION_IDLE_TIME, self.idle_time)
        if self.ret < 1:
            raise ValueError("Error idle time: " + str(self.ret))

        #Wait until all parameters are set before starting the transmission (this can take up to 120ms)
        time.sleep(0.12)

    def transfer_profile(self):
        self.ret = llt.transfer_profiles(self.hLLT, llt.TTransferProfileType.NORMAL_TRANSFER, 1)
        if self.ret < 1:
            raise ValueError("Error starting transfer profiles: " + str(self.ret))


        while(noProfileReceived):
            self.ret = llt.get_actual_profile(self.hLLT, self.profile_buffer, len(self.profile_buffer), llt.TProfileConfig.PROFILE,
                                        ct.byref(self.lost_profiles))
            if self.ret != len(self.profile_buffer):
                if (self.ret == llt.ERROR_PROFTRANS_NO_NEW_PROFILE):
                    time.sleep((self.idle_time+self.exposure_time)/100000)
                    continue
                else:
                    raise ValueError("Error get profile buffer data: " + str(self.ret))
                    noProfileReceived = False
            else:
                print("Profile received")
                noProfileReceived = False


        self.ret = llt.convert_profile_2_values(self.hLLT, self.profile_buffer, self.resolution, llt.TProfileConfig.PROFILE, self.scanner_type, 0, 1,
                                        self.null_ptr_short, self.intensities, self.null_ptr_short, self.x, self.z, self.null_ptr_int, self.null_ptr_int)
        if self.ret & llt.CONVERT_X is 0 or self.ret & llt.CONVERT_Z is 0 or self.ret & llt.CONVERT_MAXIMUM is 0:
            raise ValueError("Error converting data: " + str(self.ret))

        # Output of profile count
        for i in range(16):
            self.timestamp[i] = self.profile_buffer[self.resolution * 64 - 16 + i]

        llt.timestamp_2_time_and_count(self.timestamp, ct.byref(self.shutter_opened), ct.byref(self.shutter_closed), ct.byref(self.profile_count))

    def close(self):
        # Stop transmission
        self.ret = llt.transfer_profiles(self.hLLT, llt.TTransferProfileType.NORMAL_TRANSFER, 0)
        if ret < 1:
            raise ValueError("Error stopping transfer profiles: " + str(ret))

        # Disconnect
        ret = llt.disconnect(self.hLLT)
        if ret < 1:
            raise ConnectionAbortedError("Error while disconnect: " + str(ret))

        # Delete
        ret = llt.del_device(self.hLLT)
        if ret < 1:
            raise ConnectionAbortedError("Error while delete: " + str(ret))
        
    def main(self):
        self.initialize()
        self.transfer_profile()
        self.close()
