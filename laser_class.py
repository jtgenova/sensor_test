import pyllt as llt
import ctypes as ct
import numpy as np
import matplotlib.pyplot as plt
import time

class scanControl:
    def __init__(self):
        none = 0
    
    def initialize(self):
        self.hLLT = llt.create_llt_device(llt.TInterfaceType.INTF_TYPE_ETHERNET)
        time.sleep(0.2) 
        
        available_interfaces = (ct.c_uint*6)()

        llt.get_device_interfaces_fast(self.hLLT, available_interfaces, len(available_interfaces))
        llt.set_device_interface(self.hLLT, available_interfaces[0], 0)
        time.sleep(0.2)
    
    def connect(self):
        # connect sensor
        llt.connect(self.hLLT)
        time.sleep(0.2)

         # check which sensor is connected
        self.scanner_type = ct.c_int(0)
        llt.get_llt_type(self.hLLT, ct.byref(self.scanner_type))
        time.sleep(0.2)

        # check resolutions are supported and set highest resolution
        available_resolutions = (ct.c_uint*4)()
        llt.get_resolutions(self.hLLT, available_resolutions, len(available_resolutions))
        self.resolution = available_resolutions[0]
        llt.set_resolution(self.hLLT, self.resolution)
        time.sleep(0.2)

        # set profile config
        llt.set_profile_config(self.hLLT, llt.TProfileConfig.PROFILE)
        time.sleep(0.2)
        

    def transfer_profile(self):
        # start profile transfer
        llt.transfer_profiles(self.hLLT, llt.TTransferProfileType.NORMAL_TRANSFER, 1)
        time.sleep(0.2)
        
        # Allocate correctly sized buffer array and fetch the lastest received profile raw data from the internal receiving buffer.
        profile_buffer = (ct.c_ubyte*(self.resolution*64))()
        lost_profiles = ct.c_int()
        llt.get_actual_profile(self.hLLT, profile_buffer, len(profile_buffer), llt.TProfileConfig.PROFILE, ct.byref(lost_profiles))
        time.sleep(0.2)

        # stop profile transfer       
        # llt.transfer_profiles(self.hLLT, TTransferProfileType.NORMAL_TRANSFER, 0)
        llt.transfer_profiles(self.hLLT, 0, 0)
        time.sleep(0.2)

        x = (ct.c_double * self.resolution)()
        z = (ct.c_double * self.resolution)()
        intensities = (ct.c_ushort * self.resolution)()

        snull = ct.POINTER(ct.c_ushort)()
        inull = ct.POINTER(ct.c_uint)()

        llt.convert_profile_2_values(self.hLLT, profile_buffer, self.resolution, llt.TProfileConfig.PROFILE, self.scanner_type, 0, 1,
                                            snull, intensities, snull, x, z, inull, inull)

        time.sleep(0.2)

        llt.disconnect(self.hLLT)
        time.sleep(0.2)

        return x, z
    

    def plot(self, x, z):
        plt.figure(facecolor='white')
        plt.grid()
        plt.xlabel('x')
        plt.ylabel('z')
        plt.xlim(-200, 200)
        plt.ylim(150, 450)
        plt.plot(x, z, 'r.', label="z", lw=1)
        plt.show()

    
    def main(self):
        self.initialize()
        self.connect()
        x, z = self.transfer_profile()
        print(min(np.array(x)), max(np.array(x)))
        print(min(np.array(z)), max(np.array(z)))
        
        self.plot(x, z)
        
        



if __name__ == "__main__":
    laser = scanControl()
    laser.main()