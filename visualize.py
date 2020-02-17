import numpy as np
import pptk
import open3d as o3d
import scipy.io as sio
import os

result_dir = './results/faust/'
if __name__ == '__main__':
    for root, _, files in os.walk(result_dir):
        for file in files:
            print(file)
            file_path = os.path.join(root, file)
            m = sio.loadmat(file_path)
            x, y, Q = m['x'], m['y'], m['Q']
            Q = np.argmax(Q, axis=0)
            colors = (y + 1) / 2
            x_o3d, y_o3d = o3d.PointCloud(), o3d.PointCloud()
            x_o3d.points, y_o3d.points = o3d.Vector3dVector(x+2), o3d.Vector3dVector(y)
            x_o3d.colors, y_o3d.colors = o3d.Vector3dVector(colors[Q]), o3d.Vector3dVector(colors)
            o3d.draw_geometries([x_o3d, y_o3d])

            # v = pptk.viewer(np.concatenate([y, x + 2.], axis=0), np.concatenate([colors, colors[Q]]), axis=0)
            # v.set(point_size=0.02)
