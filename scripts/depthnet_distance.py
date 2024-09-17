import sys
import argparse
import numpy as np

from jetson_inference import depthNet
from jetson_utils import videoSource, videoOutput, cudaOverlay, cudaDeviceSynchronize, Log, cudaToNumpy

from depthnet_utils import depthBuffers

def calculate_distance(depth_map, x, y):
    """
    Calculate the distance to a point in the depth map.
    
    :param depth_map: The depth map from depthNet
    :param x: X-coordinate of the point
    :param y: Y-coordinate of the point
    :return: Distance in meters
    """
    depth = depth_map[y, x]
    # Assuming the depth is already in meters. If not, you may need to convert it.
    return depth

# parse the command line
parser = argparse.ArgumentParser(description="Mono depth estimation on a video/image stream using depthNet DNN.", 
                                 formatter_class=argparse.RawTextHelpFormatter, 
                                 epilog=depthNet.Usage() + videoSource.Usage() + videoOutput.Usage() + Log.Usage())

parser.add_argument("input", type=str, default="", nargs='?', help="URI of the input stream")
parser.add_argument("output", type=str, default="", nargs='?', help="URI of the output stream")
parser.add_argument("--network", type=str, default="fcn-mobilenet", help="pre-trained model to load, see below for options")
parser.add_argument("--visualize", type=str, default="input,depth", help="visualization options (can be 'input' 'depth' 'input,depth'")
parser.add_argument("--depth-size", type=float, default=1.0, help="scales the size of the depth map visualization, as a percentage of the input size (default is 1.0)")
parser.add_argument("--filter-mode", type=str, default="linear", choices=["point", "linear"], help="filtering mode used during visualization, options are:\n  'point' or 'linear' (default: 'linear')")
parser.add_argument("--colormap", type=str, default="viridis-inverted", help="colormap to use for visualization (default is 'viridis-inverted')",
                                  choices=["inferno", "inferno-inverted", "magma", "magma-inverted", "parula", "parula-inverted", 
                                           "plasma", "plasma-inverted", "turbo", "turbo-inverted", "viridis", "viridis-inverted"])

try:
    args = parser.parse_known_args()[0]
except:
    print("")
    parser.print_help()
    sys.exit(0)

# load the segmentation network
net = depthNet(args.network, sys.argv)

# create buffer manager
buffers = depthBuffers(args)

# create video sources & outputs
input = videoSource(args.input, argv=sys.argv)
output = videoOutput(args.output, argv=sys.argv)

# process frames until EOS or the user exits
while True:
    # capture the next image
    img_input = input.Capture()

    if img_input is None: # timeout
        continue
        
    # allocate buffers for this size image
    buffers.Alloc(img_input.shape, img_input.format)

    # process the mono depth and visualize
    net.Process(img_input, buffers.depth, args.colormap, args.filter_mode)

    # Get the depth map as a numpy array
    depth_map = cudaToNumpy(buffers.depth)

    # Example: Calculate distance to the center of the image
    center_x, center_y = img_input.width // 2, img_input.height // 2
    distance = calculate_distance(depth_map, center_x, center_y)
    #print(f"Distance to center: {distance:.2f} meters")
    print("distance",distance)
    # composite the images
    if buffers.use_input:
        cudaOverlay(img_input, buffers.composite, 0, 0)
        
    if buffers.use_depth:
        cudaOverlay(buffers.depth, buffers.composite, img_input.width if buffers.use_input else 0, 0)

    # render the output image
    output.Render(buffers.composite)

    # update the title bar
    output.SetStatus("{:s} | {:s} | Network {:.0f} FPS".format(args.network, net.GetNetworkName(), net.GetNetworkFPS()))

    # print out performance info
    cudaDeviceSynchronize()
    net.PrintProfilerTimes()

    # exit on input/output EOS
    if not input.IsStreaming() or not output.IsStreaming():
        break