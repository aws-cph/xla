import sys
import unittest
import torch
from torch.distributed.tensor.placement_types import Shard, Replicate
import torch_xla.runtime as xr
import torch_xla.distributed.spmd as xs
import torch_xla
import numpy as np
import test_xla_sharding_base


class DTensorRedistributeTest(test_xla_sharding_base.XlaShardingTest):

  @classmethod
  def setUpClass(cls):
    super().setUpClass()

  @unittest.skipIf(xr.global_runtime_device_count() < 2,
                   "At least 2 devices needed for redistribute test")
  def test_redistribute_method(self):
    device_count = xr.global_runtime_device_count()

    # XLA Mesh requires all devices, generate shapes that use all devices
    mesh_shapes = [(device_count,)]  # 1D using all devices
    if device_count >= 4 and device_count % 2 == 0:
      mesh_shapes.append((2, device_count // 2))  # 2D using all devices
    if device_count >= 4 and device_count % 4 == 0:
      mesh_shapes.append((4, device_count // 4))  # Another 2D option

    for mesh_shape in mesh_shapes:
      with self.subTest(mesh_shape=mesh_shape):
        device_ids = np.array(range(device_count))
        mesh = xs.Mesh(device_ids, mesh_shape)

        tensor = torch.randn(8, 16).to('xla')
        sharded_tensor = xs.mark_sharding(tensor, mesh, (0, None))

        # Generate placement combinations dynamically
        placement_types = [Replicate(), Shard(0), Shard(1)]

        for placement in placement_types:
          if len(mesh_shape) == 1:  # 1D
            placements = [placement]
            # For 1D mesh, placement describes how tensor dims map to mesh dim 0
            if isinstance(placement, Shard):
              expected_spec = [None] * 2
              expected_spec[placement.dim] = 0
              expected_spec = tuple(expected_spec)
            else:
              expected_spec = (None, None)
          else:
            # Test combinations for 2D meshes
            for second_placement in placement_types:
              placements = [placement, second_placement]
              # For 2D mesh, convert placements to partition_spec
              expected_spec = [None] * 2
              if isinstance(placement, Shard):
                expected_spec[placement.dim] = 0
              if isinstance(second_placement, Shard):
                expected_spec[second_placement.dim] = 1
              expected_spec = tuple(expected_spec)

              redistributed = sharded_tensor.redistribute(mesh, placements)
              self.assertEqual(redistributed.partition_spec, expected_spec)

              hlo = torch_xla._XLAC._get_xla_tensors_hlo(
                  [redistributed.global_tensor])
              self.assertIn('sharding=', hlo)

            continue

          redistributed = sharded_tensor.redistribute(mesh, placements)
          self.assertEqual(redistributed.partition_spec, expected_spec)

          hlo = torch_xla._XLAC._get_xla_tensors_hlo(
              [redistributed.global_tensor])
          self.assertIn('sharding=', hlo)

  @unittest.skipIf(xr.global_runtime_device_count() < 4,
                   "At least 4 devices needed for async redistribute test")
  def test_redistribute_async(self):
    device_count = xr.global_runtime_device_count()
    mesh_shape = (2, device_count // 2)
    device_ids = np.array(range(device_count))
    mesh = xs.Mesh(device_ids, mesh_shape)

    tensor = torch.randn(8, 16).to('xla')
    sharded_tensor = xs.mark_sharding(tensor, mesh, (0, None))

    # Test async redistribute
    placements = [Replicate(), Shard(0)]
    redistributed = sharded_tensor.redistribute(mesh, placements, async_op=True)
    # [Replicate(), Shard(0)] means tensor dim 0 sharded on mesh dim 1
    self.assertEqual(redistributed.partition_spec, (1, None))

    # Verify HLO contains sharding annotations
    hlo = torch_xla._XLAC._get_xla_tensors_hlo([redistributed.global_tensor])
    self.assertIn('sharding=', hlo)


if __name__ == '__main__':
  test = unittest.main()
  sys.exit(0 if test.result.wasSuccessful() else 1)
