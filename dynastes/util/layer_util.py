import inspect


def call_masked(layer, inputs, training=None, mask=None, **kwargs):
    if hasattr(layer, 'call_masked'):
        return layer.call_masked(inputs, training=training, mask=mask, **kwargs)
    else:
        if 'mask' in inspect.signature(layer.call).parameters:
            out = layer(inputs, training=training, mask=mask, **kwargs)
        else:
            out = layer(inputs, training=training, **kwargs)
        out_mask = compute_mask_if_possible(layer, inputs, mask=mask)
        return out, out_mask


def compute_mask_if_possible(layer, inputs, mask=None):
    if layer.supports_masking:
        out_mask = layer.compute_mask(inputs, mask)
    else:
        out_mask = mask
    return out_mask
