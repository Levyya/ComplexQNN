import torch
from torch.nn.functional import max_pool2d


def _retrieve_elements_from_indices(tensor, indices):
    flattened_tensor = tensor.flatten(start_dim=-2)
    output = flattened_tensor.gather(dim=-1, index=indices.flatten(start_dim=-2)).view_as(indices)
    return output


def abs_fn(x):
    return torch.sqrt(x.real ** 2 + x.imag ** 2)


def mine_complex_max_pool2d(input, kernel_size, stride=None, padding=0,
                            dilation=1, ceil_mode=False, return_indices=False):
    """
    Perform complex max pooling by selecting on the absolute value on the complex values.
    """
    abs_value = abs_fn(input)
    absolute_value, indices = max_pool2d(
        abs_value,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        ceil_mode=ceil_mode,
        return_indices=True
    )
    # performs the selection on the absolute values
    absolute_value = absolute_value.type(torch.complex64)
    # retrieve the corresponding phase value using the indices
    # unfortunately, the derivative for 'angle' is not implemented
    angle = torch.atan2(input.imag, input.real)
    # get only the phase values selected by max pool
    angle = _retrieve_elements_from_indices(angle, indices)
    return absolute_value \
           * (torch.cos(angle).type(torch.complex64) + 1j * torch.sin(angle).type(torch.complex64))


def binary_acc(preds, y):
    preds = torch.round(torch.sigmoid(preds))
    correct = torch.eq(preds, y).float()
    acc = correct.sum() / len(correct)
    return acc