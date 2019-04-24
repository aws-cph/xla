#ifndef TENSORFLOW_COMPILER_XLA_RPC_XLA_UTIL_H_
#define TENSORFLOW_COMPILER_XLA_RPC_XLA_UTIL_H_

#include <string>

#include "tensorflow/compiler/xla/client/xla_computation.h"
#include "tensorflow/compiler/xla/service/hlo.pb.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/core/lib/gtl/array_slice.h"

namespace xla {
namespace util {

// Creates the HLO module which is generated by the input PB message.
StatusOr<std::unique_ptr<HloModule>> CreateModuleFromProto(
    const HloModuleProto& proto,
    const DebugOptions& debug_options = DebugOptions());

// Returns a textual representation of the input XLA computation.
StatusOr<string> GetComputationHloText(const XlaComputation& computation);

// Checks whether an action on the given computation generated an error, and if
// that was the case, emit error and computations HLO text.
void CheckComputationStatus(
    const Status& status,
    tensorflow::gtl::ArraySlice<const XlaComputation* const> computations);

size_t ShapeHash(const Shape& shape);

}  // namespace util
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_RPC_XLA_UTIL_H_
