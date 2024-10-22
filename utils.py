import numpy as np
import torch
from torch.nn import functional as F

def pitchyaw_to_vector(a):
    """
    expect to get a tensor [num_gazes, 2]
    """
    if a.shape[1] == 2:
        sin = torch.sin(a)
        cos = torch.cos(a)
        return torch.stack([cos[:, 0] * sin[:, 1], sin[:, 0], cos[:, 0] * cos[:, 1]], dim=1)
    elif a.shape[1] == 3:
        return F.normalize(a)
    else:
        raise ValueError('Do not know how to convert tensor of size %s' % a.shape)


def to_screen_coordinates2(origin, direction, rotation, inv_camera_transformation,ppm):

    # De-rotate gaze vector
    inv_rotation = torch.transpose(rotation, 1, 2)
    direction = direction.reshape(-1, 3, 1)
    direction = torch.matmul(inv_rotation, direction)

    # Transform values
    direction = apply_rotation(inv_camera_transformation, direction)
    origin = apply_transformation(inv_camera_transformation, origin)

    # Intersect with z = 0
    recovered_target_2D = get_intersect_with_zero(origin, direction)
    PoG_mm = recovered_target_2D

    # Convert back from mm to pixels
    ppm_w = ppm[:, 0]
    ppm_h = ppm[:, 1]
    PoG_px = torch.stack([
        torch.clamp(recovered_target_2D[:, 0] * ppm_w,
                    0.0, float(config.actual_screen_size[0])),
        torch.clamp(recovered_target_2D[:, 1] * ppm_h,
                    0.0, float(config.actual_screen_size[1]))
    ], axis=-1)

    return PoG_mm, PoG_px