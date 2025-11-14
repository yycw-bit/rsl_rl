import sys
import os
current_dir = os.path.dirname(__file__)
print(current_dir)
print(os.pardir)
sys.path.append("/home/bit/IsaacLab/source")
print(sys.path)
import isaaclab

print(isaaclab.__file__)