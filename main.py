import pyzed.sl as sl
import sys
from vedo import *

def main():
    # Create a ZEDCamera object
    zed = sl.Camera()

    # Create a InitParameters object and set configuration parameters
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD1080  # Use HD1080 video mode for higher resolution
    # Use a right-handed Y-up coordinate system
    init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP
    init_params.coordinate_units = sl.UNIT.METER  # Set units in meters

    # Open the camera
    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        print("Camera Open : "+repr(err)+". Exit program.")
        exit()

    # Enable positional tracking with default parameters.
    # Positional tracking needs to be enabled before using spatial mapping
    py_transform = sl.Transform()
    tracking_parameters = sl.PositionalTrackingParameters(_init_pos=py_transform)
    err = zed.enable_positional_tracking(tracking_parameters)
    if err != sl.ERROR_CODE.SUCCESS:
        print("Enable positional tracking : "+repr(err)+". Exit program.")
        zed.close()
        exit()

    # Enable spatial mapping
    mapping_parameters = sl.SpatialMappingParameters(map_type=sl.SPATIAL_MAP_TYPE.MESH)
    mapping_parameters.resolution_meter = 0.03  # Set resolution to 3cm for higher detail
    err = zed.enable_spatial_mapping(mapping_parameters)
    if err != sl.ERROR_CODE.SUCCESS:
        print("Enable spatial mapping : "+repr(err)+". Exit program.")
        zed.close()
        exit(1)

    # Grab data during 500 frames
    i = 0
    mesh = sl.Mesh()  # Create a Mesh object
    runtime_parameters = sl.RuntimeParameters()

    while i < 500:
        # For each new grab, mesh data is updated
        if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
            # In the background, spatial mapping will use newly retrieved images, depth and pose to update the mesh
            mapping_state = zed.get_spatial_mapping_state()
            sys.stdout.write("Images captured: {0} / 500 || {1} \033[K\r".format(i, mapping_state))
            sys.stdout.flush()            

            # Save one of the images as a JPEG file for texture mapping
            image = sl.Mat()
            zed.retrieve_image(image, sl.VIEW.LEFT)  # Retrieve the left image
            image.write("current_frame.jpg")  # Save the left image as "current_frame.jpg"

            i += 1

    print("\n")

    # Extract, filter and save the mesh in an obj file
    print("Extracting Mesh...\n")
    err = zed.extract_whole_spatial_map(mesh)
    print(repr(err))
    print("Filtering Mesh...\n")
    mesh.filter(sl.MeshFilterParameters())  # Filter the mesh (remove unnecessary vertices and faces)
    print("Saving Mesh...\n")
    mesh.save("mesh.obj")

    # Disable tracking and mapping and close the camera
    zed.disable_spatial_mapping()
    zed.disable_positional_tracking()
    zed.close()

    # Load the mesh from obj file
    mesh = Mesh("mesh.obj")

    # Apply texture from the captured image
    mesh.texture("current_frame.jpg")

    # Display the mesh
    mesh.show()

if __name__ == "__main__":
    main()
