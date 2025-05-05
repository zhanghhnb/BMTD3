import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/ubuntu20/CODE/code/BMTD3_CODE/install/turtlebot3_drl'
