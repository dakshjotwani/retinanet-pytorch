import numpy as np
import torch
import torch.nn as nn

def_pyramid_lvls = [3, 4, 5, 6, 7]
def_ratios       = torch.tensor([[1, 1, 2],
                                 [2, 1, 1]],
                                 dtype=torch.float32)
def_scales       = torch.tensor([2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)], dtype=torch.float32)

def build_anchor_boxes(pyramid_levels=def_pyramid_lvls, ratios=def_ratios, scales=def_scales):
    num_ratios = ratios.shape[1]
    num_scales = scales.shape[0]
    repeated_ratios = ratios.repeat_interleave(num_scales, dim=1)
    repeated_scales = scales.repeat(num_ratios)

    out = []

    # np.tile -> torch.repeat, np.repeat -> torch.repeat_interleave()
    for idx, pyramid_lvl in enumerate(pyramid_levels):
        base_size = 2 ** (pyramid_lvl + 2)

        unit_sides = torch.sqrt(((base_size * repeated_scales) ** 2) / (repeated_ratios[0, :] * repeated_ratios[1, :]))

        # multiply unit sides by respective ratios
        anchor_box_shapes = repeated_ratios * unit_sides
        out.append(anchor_box_shapes)

    return out


def get_box_centers(input_img_height, input_img_width, pyramid_levels=def_pyramid_lvls):
    img_res = torch.tensor([[input_img_height, input_img_width]], dtype=torch.float32)
    out_shapes = torch.cat([(img_res + 2 ** x - 1) // (2 ** x) for x in pyramid_levels])

    ret = []
    
    for i, P_lvl in enumerate(pyramid_levels):
        H, W = out_shapes[i]

        scale = 2 ** P_lvl
        x_ctr_coords = scale * torch.arange(0.5, W)
        y_ctr_coords = scale * torch.arange(0.5, H)
        
        ret.append((x_ctr_coords, y_ctr_coords))

        #print(P_lvl)
        #print(x_ctr_coords.shape)
        #print(y_ctr_coords.shape)
        #print('')

    return ret


class Anchors(nn.Module):
    def __init__(self):
        self.pyramid_levels = [3, 4, 5, 6, 7]
        self.strides = [2 ** x for x in self.pyramid_levels]
        self.sizes = [2 ** (x + 2) for x in self.pyramid_levels]
        self.ratios = torch.tensor([[1, 1, 2],
                                    [2, 1, 1]], dtype=torch.float32)
        self.scales = torch.tensor([2**0, 2**(1.0 / 3.0), 2**(2.0 / 3.0)], dtype=torch.float32)

        self.num_anchors = len(self.ratios) * len(self.scales)





def generate_anchors(base_size=16, ratios=None, scales=None):
    """
    Generate anchor (reference) windows by enumerating aspect ratios X
    scales w.r.t. a reference window.
    """

    if ratios is None:
        ratios = np.array([0.5, 1, 2])

    if scales is None:
        scales = np.array([2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)])

    num_anchors = len(ratios) * len(scales)

    # initialize output anchors
    anchors = np.zeros((num_anchors, 4))

    # scale base_size
    anchors[:, 2:] = base_size * np.tile(scales, (2, len(ratios))).T

    # compute areas of anchors
    areas = anchors[:, 2] * anchors[:, 3]

    # correct for ratios
    anchors[:, 2] = np.sqrt(areas / np.repeat(ratios, len(scales)))
    anchors[:, 3] = anchors[:, 2] * np.repeat(ratios, len(scales))

    # transform from (x_ctr, y_ctr, w, h) -> (x1, y1, x2, y2)
    anchors[:, 0::2] -= np.tile(anchors[:, 2] * 0.5, (2, 1)).T
    anchors[:, 1::2] -= np.tile(anchors[:, 3] * 0.5, (2, 1)).T

    return anchors

def compute_shape(image_shape, pyramid_levels):
    """Compute shapes based on pyramid levels.

    :param image_shape:
    :param pyramid_levels:
    :return:
    """
    image_shape = np.array(image_shape[:2])
    image_shapes = [(image_shape + 2 ** x - 1) // (2 ** x) for x in pyramid_levels]
    return image_shapes


def anchors_for_shape(
    image_shape,
    pyramid_levels=None,
    ratios=None,
    scales=None,
    strides=None,
    sizes=None,
    shapes_callback=None,
):

    image_shapes = compute_shape(image_shape, pyramid_levels)

    # compute anchors over all pyramid levels
    all_anchors = np.zeros((0, 4))
    for idx, p in enumerate(pyramid_levels):
        anchors         = generate_anchors(base_size=sizes[idx], ratios=ratios, scales=scales)
        shifted_anchors = shift(image_shapes[idx], strides[idx], anchors)
        all_anchors     = np.append(all_anchors, shifted_anchors, axis=0)

    return all_anchors


def shift(shape, stride, anchors):
    shift_x = (np.arange(0, shape[1]) + 0.5) * stride
    shift_y = (np.arange(0, shape[0]) + 0.5) * stride

    shift_x, shift_y = np.meshgrid(shift_x, shift_y)

    shifts = np.vstack((
        shift_x.ravel(), shift_y.ravel(),
        shift_x.ravel(), shift_y.ravel()
    )).transpose()

    # add A anchors (1, A, 4) to
    # cell K shifts (K, 1, 4) to get
    # shift anchors (K, A, 4)
    # reshape to (K*A, 4) shifted anchors
    A = anchors.shape[0]
    K = shifts.shape[0]
    all_anchors = (anchors.reshape((1, A, 4)) + shifts.reshape((1, K, 4)).transpose((1, 0, 2)))
    all_anchors = all_anchors.reshape((K * A, 4))

    return all_anchors

