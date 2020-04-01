#include <stdio.h>
#include <type_traits>

#include "Halide.h"

using namespace Halide;

bool error_occurred;
void halide_error(void *user_context, const char *msg) {
    printf("%s\n", msg);
    error_occurred = true;
}

int main(int argc, char **argv) {
    Func f;
    Var x;
    f(x) = Tuple(x, sin(x));

    // These should be the same size
    Buffer<int> x_out(100);
    Buffer<float> sin_x_out(101);

    f.set_error_handler(&halide_error);
    error_occurred = false;

    Realization r(x_out, sin_x_out);
    f.realize(r);

    if (!error_occurred) {
        printf("There should have been an error\n");
        return -1;
    }

    return 0;
}
