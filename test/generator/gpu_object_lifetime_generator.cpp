#include <stdint.h>
#include <memory>

#include "Halide.h"

namespace halide_register_generator {
struct halide_global_ns;
}  // namespace halide_register_generator

namespace {

class GpuObjectLifetime : public Halide::Generator<GpuObjectLifetime> {
public:
    Output<Buffer<int32_t>> output{"output", 1};

    void generate() {
        Var x;

        output(x) = x;

        Target target = get_target();
        if (target.has_gpu_feature()) {
            Var xo, xi;
            output.gpu_tile(x, xo, xi, 16);
        }
    }
};

}  // namespace

HALIDE_REGISTER_GENERATOR(GpuObjectLifetime, gpu_object_lifetime)
