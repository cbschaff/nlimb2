import numpy as np
from scipy.spatial.transform import Rotation


class Transform(object):
    def __init__(self, pos=None, quat=None, T=None):
        if T is not None:
            self.T = T
            self.R = T[:3, :3]
            self.pos = T[:3, -1]
            self.rot = Rotation.from_matrix(self.R)
            self.quat = self.rot.as_quat()
        else:
            self.pos = np.zeros(3) if pos is None else np.array(pos)
            self.quat = np.array([0., 0., 0., 1.]) if quat is None else np.array(quat)
            self.rot = Rotation.from_quat(self.quat)
            self.R = self.rot.as_matrix()
            self.T = np.eye(4)
            self.T[:3, :3] = self.R
            self.T[:3, -1] = self.pos

    def adjoint(self):
        def _skew(p):
            return np.array([
                [0, -p[2], p[1]],
                [p[2], 0, -p[0]],
                [-p[1], p[0], 0],
            ])

        adj = np.zeros((6, 6))
        adj[:3, :3] = self.R
        adj[3:, 3:] = self.R
        adj[3:, :3] = _skew(self.pos).dot(self.R)
        return adj

    def inverse(self):
        T = np.eye(4)
        T[:3, :3] = self.R.T
        T[:3, -1] = -self.R.T.dot(self.pos)
        return Transform(T=T)

    def __call__(self, x):
        if isinstance(x, Transform):
            return Transform(T=self.T.dot(x.T))
        else:
            # check for different input forms
            one_dim = len(x.shape) == 1
            homogeneous = x.shape[-1] == 4
            if one_dim:
                x = x[None]
            if not homogeneous:
                x_homo = np.ones((x.shape[0], 4))
                x_homo[:, :3] = x
                x = x_homo

            # transform points
            x = self.T.dot(x.T).T

            # create output to match input form
            if not homogeneous:
                x = x[:, :3]
            if one_dim:
                x = x[0]
            return x
