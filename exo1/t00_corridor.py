import mujoco
from mujoco import viewer
from pathlib import Path
import numpy as np

script_dir = Path(__file__).parent
xml_path = "corridor_3x100.xml"

#
### Load the model, make two MjData's. ###
#
m = mujoco.MjModel.from_xml_path(str(xml_path))

d = mujoco.MjData(m)  # data allocation


def print_camera_info(v):
    cam = v.cam
    print("=== Camera Parameters ===")
    print(f"Type: {cam.type}")
    print(f"Lookat: [{cam.lookat[0]:.2f}, {cam.lookat[1]:.2f}, {cam.lookat[2]:.2f}]")
    print(f"Azimuth: {cam.azimuth:.1f}°")
    print(f"Elevation: {cam.elevation:.1f}°")
    print(f"Distance: {cam.distance:.2f}")
    print("========================")


# Global viewer reference for keyboard callback
viewer_instance = None
display_camera_info = False
quit_requested = False


def key_callback(keycode):
    global viewer_instance, display_camera_info, quit_requested
    # print (f"Key pressed: {keycode}")
    if keycode == ord("q") or keycode == ord("Q"):
        print("\nQuitting...")
        quit_requested = True
    if keycode == ord("c") or keycode == ord("C"):
        display_camera_info = not display_camera_info
    elif keycode == 256:  # ESC key
        print("\nESC pressed - Quitting...")
        quit_requested = True
    ## Ex.02: Print camera info in main loop when 'c' or 'C' is pressed


with viewer.launch_passive(m, d, key_callback=key_callback) as v:
    viewer_instance = v
    print("Controls:")
    print("  'c' or 'C': Print camera parameters")
    print("  'q' or 'Q': Quit")
    print("  ESC: Quit")

    # Configure camera parameters with your good settings
    cam = v.cam

    # Camera parameters
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE  # Free camera
    cam.fixedcamid = -1  # Not using fixed camera

    # Ex.03: it would be better if we could see the inside of the corridor at start
    ## Set your good camera position and orientation parameters here
    #
    cam.azimuth = 1.01171875
    cam.elevation = -16.6640625
    cam.lookat = np.array([ 1.55633679e-04, -4.88295545e-02,  1.05485916e+00])

    while v.is_running():

        if quit_requested:
            break

        ## Ex.02: Print camera info when requested
        #
        if display_camera_info:
            #
            print(cam)
            #
            display_camera_info = False

        mujoco.mj_step(m, d)


        v.sync()
