#include "torch_xla/csrc/ops/device_data.h"

#include <sstream>

#include "torch_xla/csrc/lowering_context.h"
#include "torch_xla/csrc/ops/xla_ops.h"
#include "torch_xla/csrc/runtime/runtime.h"
#include "torch_xla/csrc/tensor_util.h"

namespace torch_xla {

DeviceData::DeviceData(std::shared_ptr<torch::lazy::BackendData> data)
    : XlaNode(xla_device_data,
              std::dynamic_pointer_cast<runtime::ComputationClient::Data>(data)
                  ->shape(),
              /*num_outputs=*/1,
              /*hash_seed=*/(uint32_t)101),
      data_(std::move(data)) {
  std::optional<xla::OpSharding> op_sharding =
      torch_xla::runtime::GetComputationClientOrDie()->GetDataSharding(
          std::dynamic_pointer_cast<runtime::ComputationClient::Data>(data_));
  if (op_sharding.has_value()) {
    // DeviceData Node only has 1 output.
    SetSharding(op_sharding.value(), 0);
  }
}

std::string DeviceData::ToString() const {
  std::stringstream ss;
  ss << XlaNode::ToString() << ", device=" << data_->device();
  return ss.str();
}

torch::lazy::NodePtr DeviceData::Clone(torch::lazy::OpList operands) const {
  return torch_xla::MakeNode<DeviceData>(data_);
}

XlaOpVector DeviceData::Lower(LoweringContext* loctx) const {
  return ReturnOp(loctx->GetParameter(data_, unbounded_dynamic_dims_), loctx);
}

DeviceData* DeviceData::Cast(const torch::lazy::Node* node) {
  return torch_xla::NodeCast<DeviceData>(node, xla_device_data);
}

}  // namespace torch_xla
