import numpy as np
from scipy.spatial.transform import Rotation as R

base_frame = [0.0, 0.0, 0.0, np.pi / np.sqrt(2), -np.pi / np.sqrt(2), 0.0]

base_rot = R.from_rotvec(base_frame[3:6], degrees=False)

op_rot = R.from_euler('x', -10, degrees=True)

print((base_rot * op_rot).as_rotvec())

new_frame = [0.0, 0.0, 0.0] + (base_rot * op_rot).as_rotvec().tolist()

print(new_frame)