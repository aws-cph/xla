import torch
import torch_xla


# TODO(lsy323): Register as a torch op, cannot do that because parameter
# `ScalarType[] output_dtypes` in the op schema has some problem.
def stablehlo_custom_call(args,
                          call_target,
                          output_shapes,
                          output_dtypes,
                          has_side_effect=False,
                          backend_config="",
                          api_version=0,
                          frontend_attributes=None):
  frontend_attributes = frontend_attributes or {}
  res = torch_xla._XLAC._xla_custom_call(args, call_target, output_shapes,
                                         output_dtypes, has_side_effect,
                                         backend_config, api_version,
                                         frontend_attributes)
  if len(output_shapes) == 1:
    return res[0]
  return res


def extract_custom_call_outputs_shape_dtype(n: torch.fx.Node):
  assert 'val' in n.meta
  if isinstance(n.meta['val'], torch.Tensor):
    return [n.meta['val'].shape], [n.meta['val'].dtype]
  output_shape_dtype = [(t.shape,
                         t.dtype) if isinstance(t, torch.Tensor) else None
                        for t in n.meta['val']]
  assert None not in output_shape_dtype
  output_shape, output_dtype = zip(*output_shape_dtype)
  return output_shape, output_dtype


def place_to_host(a: torch.Tensor):
  return stablehlo_custom_call(
      [a],
      "annotate_device_placement", [a.shape], [a.dtype],
      has_side_effect=True,
      frontend_attributes={"_xla_buffer_placement": "pinned_host"})


def place_to_device(a: torch.Tensor):
  return stablehlo_custom_call(
      [a],
      "annotate_device_placement", [a.shape], [a.dtype],
      has_side_effect=True,
      frontend_attributes={"_xla_buffer_placement": "device"})
