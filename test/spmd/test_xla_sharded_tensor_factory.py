import unittest
import torch
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr
import torch_xla.distributed.spmd as xs
from torch_xla.distributed.spmd import zeros, ones, empty, full, randn, rand
from absl.testing import parameterized
import test_xla_sharding_base


class TestXLAShardedTensorFactory(test_xla_sharding_base.XlaShardingTest,
                                  parameterized.TestCase):

  def setUp(self):
    device_count = xr.global_runtime_device_count()
    if device_count >= 4:
      self.mesh_2d = xs.Mesh(list(range(4)), (2, 2))
      self.mesh_1d = xs.Mesh(list(range(device_count)), (device_count,))
    else:
      self.mesh_1d = xs.Mesh(list(range(device_count)), (device_count,))
      self.mesh_2d = self.mesh_1d
    xs.set_global_mesh(self.mesh_1d)
    xr.use_spmd()

  # Test tensor shapes: 0-d, 1-d, 2-d, 3-d
  @parameterized.parameters(
      ((),),  # 0-d scalar
      ((5,),),  # 1-d
      ((3, 4),),  # 2-d
      ((2, 3, 4),)  # 3-d
  )
  def test_tensor_shapes(self, shape):
    """Test factory functions with different tensor shapes."""
    for factory_fn in [zeros, ones, empty]:
      with self.subTest(factory=factory_fn.__name__):
        if len(shape) == 0:
          tensor = factory_fn()
        else:
          tensor = factory_fn(*shape)
        self.assertEqual(tensor.shape, torch.Size(shape))

  # Test tensor precisions: bf16, f32, int32
  @parameterized.parameters(torch.bfloat16, torch.float32, torch.int32)
  def test_tensor_precisions(self, dtype):
    """Test factory functions with different dtypes."""
    for factory_fn in [zeros, ones, empty]:
      with self.subTest(factory=factory_fn.__name__):
        tensor = factory_fn(2, 2, dtype=dtype)
        self.assertEqual(tensor.dtype, dtype)

  # Test DeviceMesh: 1-d, 2-d
  @parameterized.parameters(
      ((4,), (0,)),  # 1-d mesh, 1-d tensor
      ((4, 4), (0, 1))  # 2-d mesh, 2-d tensor
  )
  def test_device_mesh_shapes(self, shape, partition_spec):
    """Test with different mesh shapes."""
    device_count = xr.global_runtime_device_count()

    if len(partition_spec) == 1:
      mesh = xs.Mesh(list(range(device_count)), (device_count,))
    elif device_count >= 4:
      mesh = xs.Mesh(list(range(4)), (2, 2))
    else:
      self.skipTest("Need ≥4 devices for 2D mesh")

    tensor = zeros(*shape, mesh=mesh, partition_spec=partition_spec)
    self.assertEqual(tensor.shape, shape)

  # Test placement types: replicate, shard
  @parameterized.parameters(
      ((None, None),),  # replicated
      ((0, None),),  # shard dim 0
      ((None, 0),)  # shard dim 1
  )
  def test_placement_types(self, partition_spec):
    """Test different placement types."""
    for factory_fn in [zeros, ones, empty]:
      with self.subTest(factory=factory_fn.__name__):
        tensor = factory_fn(4, 4, partition_spec=partition_spec)
        self.assertEqual(tensor.shape, (4, 4))

  # Test all factory functions and requires_grad
  @parameterized.parameters(
      (zeros, 0.0, True), (zeros, 0.0, False), (ones, 1.0, True),
      (ones, 1.0, False), (empty, None, True), (empty, None, False),
      (randn, None, True), (randn, None, False), (rand, None, True),
      (rand, None, False))
  def test_factory_functions_and_grad(self, factory_fn, expected_value,
                                      requires_grad):
    """Test all factory functions with requires_grad parameter."""
    if factory_fn == full:
      tensor = factory_fn((2, 3), 7.5, requires_grad=requires_grad)
      expected_tensor = torch.full((2, 3), 7.5)
    else:
      tensor = factory_fn(2, 3, requires_grad=requires_grad)
      if expected_value is not None:
        if factory_fn == zeros:
          expected_tensor = torch.zeros(2, 3)
        elif factory_fn == ones:
          expected_tensor = torch.ones(2, 3)

    self.assertEqual(tensor.shape, (2, 3))
    self.assertEqual(tensor.requires_grad, requires_grad)

    if expected_value is not None and factory_fn in [zeros, ones]:
      self.assertTrue(torch.allclose(tensor.cpu(), expected_tensor))

  # Test full function separately due to different signature
  def test_full_function(self):
    """Test full function specifically."""
    f = full((2, 3), 7.5)
    self.assertEqual(f.shape, (2, 3))
    self.assertTrue(torch.allclose(f.cpu(), torch.full((2, 3), 7.5)))

    f_grad = full((2, 3), 7.5, requires_grad=True)
    self.assertTrue(f_grad.requires_grad)

  # Test XLA device requirement
  def test_xla_device_required(self):
    """Test that tensors are created on XLA device."""
    tensor = zeros(2, 2)
    self.assertTrue(tensor.device.type == 'xla')

  # Test edge cases
  def test_edge_cases(self):
    """Test edge cases and boundary conditions."""
    # Very small tensor
    tiny = zeros(1, 1)
    self.assertEqual(tiny.shape, (1, 1))

    # Zero in one dimension
    zero_dim = empty(0, 5)
    self.assertEqual(zero_dim.shape, (0, 5))

    # Large tensor (memory efficiency)
    large = zeros(10000, 100)
    self.assertEqual(large.shape, (10000, 100))

  # Test invalid inputs
  def test_invalid_inputs(self):
    """Test error handling for invalid inputs."""
    # Negative dimensions should raise error
    with self.assertRaises((ValueError, RuntimeError)):
      zeros(-1, 2)

    # Invalid dtype
    with self.assertRaises((TypeError, RuntimeError)):
      ones(2, 2, dtype="invalid")

  # Test memory efficiency
  def test_memory_efficiency(self):
    """Test that large tensors don't cause OOM."""
    try:
      # Create large tensor that would OOM if fully materialized
      large_tensor = zeros(50000, 1000, partition_spec=(0, None))
      self.assertEqual(large_tensor.shape, (50000, 1000))
    except RuntimeError as e:
      if "out of memory" in str(e).lower():
        self.fail("OOM suggests tensor was materialized")
      else:
        raise

  # Test sharding propagation with operations
  def test_sharding_propagation(self):
    """Test that sharding is preserved through operations."""
    # Unary ops
    a = zeros(4, 4, partition_spec=(0, None))
    neg_a = -a
    self.assertEqual(neg_a.shape, (4, 4))

    # Binary ops
    b = ones(4, 4, partition_spec=(0, None))
    c = a + b
    self.assertEqual(c.shape, (4, 4))

    # Multi-nary ops (if supported)
    d = torch.stack([a, b], dim=0)
    self.assertEqual(d.shape, (2, 4, 4))


if __name__ == '__main__':
  unittest.main()
