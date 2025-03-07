
import halide as hl

import simple_pystub
import complex_pystub

def _realize_and_check(f, offset = 0):
    b = hl.Buffer(hl.Float(32), [2, 2])
    f.realize(b)

    assert b[0, 0] == 3.5 + offset + 123
    assert b[0, 1] == 4.5 + offset + 123
    assert b[1, 0] == 4.5 + offset + 123
    assert b[1, 1] == 5.5 + offset + 123


def test_simple(gen):
    x, y = hl.Var(), hl.Var()
    target = hl.get_jit_target_from_environment()

    b_in = hl.Buffer(hl.UInt(8), [2, 2])
    b_in.fill(123)

    f_in = hl.Func("f")
    f_in[x, y] = x + y

    # ----------- Inputs by-position
    f = gen(target, b_in, f_in, 3.5)
    _realize_and_check(f)

    # ----------- Inputs by-name
    f = gen(target, buffer_input=b_in, func_input=f_in, float_arg=3.5)
    _realize_and_check(f)

    f = gen(target, float_arg=3.5, buffer_input=b_in, func_input=f_in)
    _realize_and_check(f)

    # ----------- Above set again, w/ GeneratorParam mixed in
    k = 42

    gp = { "offset": k }

    # (positional)
    f = gen(target, b_in, f_in, 3.5, generator_params=gp)
    _realize_and_check(f, k)

    # (keyword)
    f = gen(target, generator_params=gp, buffer_input=b_in, func_input=f_in, float_arg=3.5)
    _realize_and_check(f, k)

    f = gen(target, buffer_input=b_in, generator_params=gp, func_input=f_in, float_arg=3.5)
    _realize_and_check(f, k)

    f = gen(target, buffer_input=b_in, func_input=f_in, generator_params=gp, float_arg=3.5)
    _realize_and_check(f, k)

    f = gen(target, buffer_input=b_in, float_arg=3.5, func_input=f_in, generator_params=gp)
    _realize_and_check(f, k)

    # ----------- Test various failure modes
    try:
        # Inputs w/ mixed by-position and by-name
        f = gen(target, b_in, f_in, float_arg=3.5)
    except hl.HalideError as e:
        assert 'Cannot use both positional and keyword arguments for inputs.' in str(e)
    else:
        assert False, 'Did not see expected exception!'

    try:
        # too many positional args
        f = gen(target, b_in, f_in, 3.5, 4)
    except hl.HalideError as e:
        assert 'Expected exactly 3 positional args for inputs, but saw 4.' in str(e)
    else:
        assert False, 'Did not see expected exception!'

    try:
        # too few positional args
        f = gen(target, b_in, f_in)
    except hl.HalideError as e:
        assert 'Expected exactly 3 positional args for inputs, but saw 2.' in str(e)
    else:
        assert False, 'Did not see expected exception!'

    try:
        # Inputs that can't be converted to what the receiver needs (positional)
        f = gen(target, hl.f32(3.141592), "happy", k)
    except hl.HalideError as e:
        assert 'Input buffer_input requires an ImageParam or Buffer argument' in str(e)
    else:
        assert False, 'Did not see expected exception!'

    try:
        # Inputs that can't be converted to what the receiver needs (named)
        f = gen(target, b_in, f_in, float_arg="bogus")
    except hl.HalideError as e:
        assert 'Input float_arg requires a Param (or scalar literal) argument' in str(e)
    else:
        assert False, 'Did not see expected exception!'

    try:
        # Input specified by both pos and kwarg
        f = gen(target, b_in, f_in, 3.5, float_arg=4.5)
    except hl.HalideError as e:
        assert "Cannot use both positional and keyword arguments for inputs." in str(e)
    else:
        assert False, 'Did not see expected exception!'

    try:
        # generator_params is not a dict
        f = gen(target, b_in, f_in, 3.5, generator_params=[1, 2, 3])
    except TypeError as e:
        assert "cannot convert dictionary" in str(e)
    else:
        assert False, 'Did not see expected exception!'

    try:
        # Bad gp name
        f = gen(target, b_in, f_in, 3.5, generator_params={"foo": 0})
    except hl.HalideError as e:
        assert "has no GeneratorParam named: foo" in str(e)
    else:
        assert False, 'Did not see expected exception!'

    try:
        # Bad input name
        f = gen(target, buffer_input=b_in, float_arg=3.5, generator_params=gp, funk_input=f_in)
    except hl.HalideError as e:
        assert "Unknown input 'funk_input' specified via keyword argument." in str(e)
    else:
        assert False, 'Did not see expected exception!'

    try:
        # Bad gp name
        f = gen(target, buffer_input=b_in, float_arg=3.5, generator_params=gp, func_input=f_in, nonexistent_generator_param="wat")
    except hl.HalideError as e:
        assert "Unknown input 'nonexistent_generator_param' specified via keyword argument." in str(e)
    else:
        assert False, 'Did not see expected exception!'

def test_looplevel(gen):
    x, y = hl.Var('x'), hl.Var('y')
    target = hl.get_jit_target_from_environment()

    buffer_input = hl.Buffer(hl.UInt(8), [4, 4])
    buffer_input.fill(123)

    func_input = hl.Func("func_input")
    func_input[x, y] = x + y

    simple_compute_at = hl.LoopLevel()
    simple = gen(target, buffer_input, func_input, 3.5,
        generator_params = {
            "compute_level": simple_compute_at
        })

    computed_output = hl.Func('computed_output')
    computed_output[x, y] = simple[x, y] + 3

    simple_compute_at.set(hl.LoopLevel(computed_output, x))

    _realize_and_check(computed_output, 3)


def _make_constant_image():
    constant_image = hl.Buffer(hl.UInt(8), [32, 32, 3], 'constant_image')
    for x in range(32):
        for y in range(32):
            for c in range(3):
                constant_image[x, y, c] = x + y + c
    return constant_image

def test_complex(gen):
    constant_image = _make_constant_image()
    input = hl.ImageParam(hl.UInt(8), 3, 'input')
    input.set(constant_image)

    x, y, c = hl.Var(), hl.Var(), hl.Var()
    target = hl.get_jit_target_from_environment()

    float_arg = 1.25
    int_arg = 33

    func_input = hl.Func("func_input")
    func_input[x, y, c] = hl.u16(x + y + c)

    r = gen(target,
            typed_buffer_input=constant_image,
            untyped_buffer_input=constant_image,
            simple_input=input,
            array_input=[ input, input ],
            float_arg=float_arg,
            int_arg=[ int_arg, int_arg ],
            extra_func_input=func_input,
            generator_params = {
                "untyped_buffer_output.type": hl.UInt(8),
                "vectorize": True
            })

    # return value is a tuple; unpack separately to avoid
    # making the callsite above unreadable
    (simple_output,
        tuple_output,
        array_output,
        typed_buffer_output,
        untyped_buffer_output,
        static_compiled_buffer_output,
        scalar_output,
        extra_func_output) = r

    b = simple_output.realize([32, 32, 3], target)
    assert b.type() == hl.Float(32)
    for x in range(32):
        for y in range(32):
            for c in range(3):
                expected = constant_image[x, y, c]
                actual = b[x, y, c]
                assert expected == actual, "Expected %s Actual %s" % (expected, actual)

    b = tuple_output.realize([32, 32, 3], target)
    assert b[0].type() == hl.Float(32)
    assert b[1].type() == hl.Float(32)
    assert len(b) == 2
    for x in range(32):
        for y in range(32):
            for c in range(3):
                expected1 = constant_image[x, y, c] * float_arg
                expected2 = expected1 + int_arg
                actual1, actual2 = b[0][x, y, c], b[1][x, y, c]
                assert expected1 == actual1, "Expected1 %s Actual1 %s" % (expected1, actual1)
                assert expected2 == actual2, "Expected2 %s Actual1 %s" % (expected2, actual2)

    assert len(array_output) == 2
    for a in array_output:
        b = a.realize([32, 32], target)
        assert b.type() == hl.Int(16)
        for x in range(32):
            for y in range(32):
                expected = constant_image[x, y, 0] + int_arg
                actual = b[x, y]
                assert expected == actual, "Expected %s Actual %s" % (expected, actual)

    # TODO: Output<Buffer<>> has additional behaviors useful when a Stub
    # is used within another Generator; this isn't yet implemented since there
    # isn't yet Python bindings for Generator authoring. This section
    # of the test may need revision at that point.
    b = typed_buffer_output.realize([32, 32, 3], target)
    assert b.type() == hl.Float(32)
    for x in range(32):
        for y in range(32):
            for c in range(3):
                expected = constant_image[x, y, c]
                actual = b[x, y, c]
                assert expected == actual, "Expected %s Actual %s" % (expected, actual)

    b = untyped_buffer_output.realize([32, 32, 3], target)
    assert b.type() == hl.UInt(8)
    for x in range(32):
        for y in range(32):
            for c in range(3):
                expected = constant_image[x, y, c]
                actual = b[x, y, c]
                assert expected == actual, "Expected %s Actual %s" % (expected, actual)

    b = static_compiled_buffer_output.realize([4, 4, 1], target)
    assert b.type() == hl.UInt(8)
    for x in range(4):
        for y in range(4):
            for c in range(1):
                expected = constant_image[x, y, c] + 42
                actual = b[x, y, c]
                assert expected == actual, "Expected %s Actual %s" % (expected, actual)

    b = scalar_output.realize([], target)
    assert b.type() == hl.Float(32)
    assert b[()] == 34.25

    b = extra_func_output.realize([32, 32], target)
    assert b.type() == hl.Float(64)
    for x in range(32):
        for y in range(32):
            expected = x + y + 1
            actual = b[x, y]
            assert expected == actual, "Expected %s Actual %s" % (expected, actual)

if __name__ == "__main__":
    test_simple(simple_pystub.generate)
    test_looplevel(simple_pystub.generate)
    test_complex(complex_pystub.generate)
