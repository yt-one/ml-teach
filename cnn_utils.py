import numpy as np
from termcolor import colored

from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import concatenate
from tensorflow.keras.layers import ZeroPadding2D
from tensorflow.keras.layers import Dense


# Compare the two inputs
def comparator(learner, instructor):
    if learner == instructor:
        for a, b in zip(learner, instructor):
            if tuple(a) != tuple(b):
                print(colored("Test failed", attrs=['bold']),
                      "\n Expected value \n\n", colored(f"{b}", "green"),
                      "\n\n does not match the input value: \n\n",
                      colored(f"{a}", "red"))
                raise AssertionError("Error in test")
        print(colored("All tests passed!", "green"))

    else:
        print(colored("Test failed. Your output is not as expected output.", "red"))


# extracts the description of a given model
def summary(model):
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    result = []
    for layer in model.layers:
        descriptors = [layer.__class__.__name__, layer.output_shape, layer.count_params()]
        if (type(layer) == Conv2D):
            descriptors.append(layer.padding)
            descriptors.append(layer.activation.__name__)
            descriptors.append(layer.kernel_initializer.__class__.__name__)
        if (type(layer) == MaxPooling2D):
            descriptors.append(layer.pool_size)
            descriptors.append(layer.strides)
            descriptors.append(layer.padding)
        if (type(layer) == Dropout):
            descriptors.append(layer.rate)
        if (type(layer) == ZeroPadding2D):
            descriptors.append(layer.padding)
        if (type(layer) == Dense):
            descriptors.append(layer.activation.__name__)
        result.append(descriptors)
    return result


def datatype_check(expected_output, target_output, error):
    success = 0
    if isinstance(target_output, dict):
        for key in target_output.keys():
            try:
                success += datatype_check(expected_output[key],
                                          target_output[key], error)
            except:
                print("Error: {} in variable {}. Got {} but expected type {}".format(error,
                                                                                     key, type(target_output[key]),
                                                                                     type(expected_output[key])))
        if success == len(target_output.keys()):
            return 1
        else:
            return 0
    elif isinstance(target_output, tuple) or isinstance(target_output, list):
        for i in range(len(target_output)):
            try:
                success += datatype_check(expected_output[i],
                                          target_output[i], error)
            except:
                print("Error: {} in variable {}, expected type: {}  but expected type {}".format(error,
                                                                                                 i,
                                                                                                 type(target_output[i]),
                                                                                                 type(expected_output[
                                                                                                          i])))
        if success == len(target_output):
            return 1
        else:
            return 0

    else:
        assert isinstance(target_output, type(expected_output))
        return 1


def equation_output_check(expected_output, target_output, error):
    success = 0
    if isinstance(target_output, dict):
        for key in target_output.keys():
            try:
                success += equation_output_check(expected_output[key],
                                                 target_output[key], error)
            except:
                print("Error: {} for variable {}.".format(error,
                                                          key))
        if success == len(target_output.keys()):
            return 1
        else:
            return 0
    elif isinstance(target_output, tuple) or isinstance(target_output, list):
        for i in range(len(target_output)):
            try:
                success += equation_output_check(expected_output[i],
                                                 target_output[i], error)
            except:
                print("Error: {} for variable in position {}.".format(error, i))
        if success == len(target_output):
            return 1
        else:
            return 0

    else:
        if hasattr(target_output, 'shape'):
            np.testing.assert_array_almost_equal(target_output, expected_output)
        else:
            assert target_output == expected_output
        return 1


def shape_check(expected_output, target_output, error):
    success = 0
    if isinstance(target_output, dict):
        for key in target_output.keys():
            try:
                success += shape_check(expected_output[key],
                                       target_output[key], error)
            except:
                print("Error: {} for variable {}.".format(error, key))
        if success == len(target_output.keys()):
            return 1
        else:
            return 0
    elif isinstance(target_output, tuple) or isinstance(target_output, list):
        for i in range(len(target_output)):
            try:
                success += shape_check(expected_output[i],
                                       target_output[i], error)
            except:
                print("Error: {} for variable {}.".format(error, i))
        if success == len(target_output):
            return 1
        else:
            return 0

    else:
        if hasattr(target_output, 'shape'):
            assert target_output.shape == expected_output.shape
        return 1


def single_test(test_cases, target):
    success = 0
    for test_case in test_cases:
        try:
            if test_case['name'] == "datatype_check":
                assert isinstance(target(*test_case['input']),
                                  type(test_case["expected"]))
                success += 1
            if test_case['name'] == "equation_output_check":
                assert np.allclose(test_case["expected"],
                                   target(*test_case['input']))
                success += 1
            if test_case['name'] == "shape_check":
                assert test_case['expected'].shape == target(*test_case['input']).shape
                success += 1
        except:
            print("Error: " + test_case['error'])

    if success == len(test_cases):
        print("\033[92m All tests passed.")
    else:
        print('\033[92m', success, " Tests passed")
        print('\033[91m', len(test_cases) - success, " Tests failed")
        raise AssertionError(
            "Not all tests were passed for {}. Check your equations and avoid using global variables inside the function.".format(
                target.__name__))


def multiple_test(test_cases, target):
    success = 0
    for test_case in test_cases:
        try:
            target_answer = target(*test_case['input'])
            if test_case['name'] == "datatype_check":
                success += datatype_check(test_case['expected'], target_answer, test_case['error'])
            if test_case['name'] == "equation_output_check":
                success += equation_output_check(test_case['expected'], target_answer, test_case['error'])
            if test_case['name'] == "shape_check":
                success += shape_check(test_case['expected'], target_answer, test_case['error'])
        except:
            print("Error: " + test_case['error'])

    if success == len(test_cases):
        print("\033[92m All tests passed.")
    else:
        print('\033[92m', success, " Tests passed")
        print('\033[91m', len(test_cases) - success, " Tests failed")
        raise AssertionError(
            "Not all tests were passed for {}. Check your equations and avoid using global variables inside the function.".format(
                target.__name__))


def zero_pad_test(target):
    # Test 1
    np.random.seed(1)
    x = np.random.randn(4, 3, 3, 2)
    x_pad = target(x, 3)
    print("x.shape =\n", x.shape)
    print("x_pad.shape =\n", x_pad.shape)
    print("x[1,1] =\n", x[1, 1])
    print("x_pad[1,1] =\n", x_pad[1, 1])

    assert type(x_pad) == np.ndarray, "Output must be a np array"
    assert x_pad.shape == (4, 9, 9, 2), f"Wrong shape: {x_pad.shape} != (4, 9, 9, 2)"
    print(x_pad[0, 0:2, :, 0])
    assert np.allclose(x_pad[0, 0:2, :, 0], [[0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0]],
                       1e-15), "Rows are not padded with zeros"
    assert np.allclose(x_pad[0, :, 7:9, 1].transpose(), [[0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0]],
                       1e-15), "Columns are not padded with zeros"
    assert np.allclose(x_pad[:, 3:6, 3:6, :], x, 1e-15), "Internal values are different"

    # Test 2
    np.random.seed(1)
    x = np.random.randn(5, 4, 4, 3)
    pad = 2
    x_pad = target(x, pad)

    assert type(x_pad) == np.ndarray, "Output must be a np array"
    assert x_pad.shape == (5, 4 + 2 * pad, 4 + 2 * pad,
                           3), f"Wrong shape: {x_pad.shape} != {(5, 4 + 2 * pad, 4 + 2 * pad, 3)}"
    assert np.allclose(x_pad[0, 0:2, :, 0], [[0, 0, 0, 0, 0, 0, 0, 0],
                                             [0, 0, 0, 0, 0, 0, 0, 0]], 1e-15), "Rows are not padded with zeros"
    assert np.allclose(x_pad[0, :, 6:8, 1].transpose(), [[0, 0, 0, 0, 0, 0, 0, 0],
                                                         [0, 0, 0, 0, 0, 0, 0, 0]],
                       1e-15), "Columns are not padded with zeros"
    assert np.allclose(x_pad[:, 2:6, 2:6, :], x, 1e-15), "Internal values are different"

    print("\033[92mAll tests passed!")


def conv_single_step_test(target):
    np.random.seed(3)
    a_slice_prev = np.random.randn(5, 5, 3)
    W = np.random.randn(5, 5, 3)
    b = np.random.randn(1, 1, 1)

    Z = target(a_slice_prev, W, b)
    expected_output = np.float64(-3.5443670581382474)

    assert (type(Z) == np.float64 or type(Z) == np.float32), "You must cast the output to float"
    assert np.isclose(Z, expected_output), f"Wrong value. Expected: {expected_output} got: {Z}"

    print("\033[92mAll tests passed!")


def conv_forward_test_1(z_mean, z_0_2_1):
    test_count = 0
    z_mean_expected = 0.5511276474566768
    z_0_2_1_expected = [-2.17796037, 8.07171329, -0.5772704, 3.36286738, 4.48113645, -2.89198428, 10.99288867,
                        3.03171932]

    if np.isclose(z_mean, z_mean_expected):
        test_count = test_count + 1
    else:
        print("\033[91mFirst Test: Z's mean is incorrect. Expected:", z_mean_expected, "\nYour output:", z_mean,
              ". Make sure you include stride in your calculation\033[90m\n")

    if np.allclose(z_0_2_1, z_0_2_1_expected):
        test_count = test_count + 1
    else:
        print("\033[91mFirst Test: Z[0,2,1] is incorrect. Expected:", z_0_2_1_expected, "\nYour output:", z_0_2_1,
              "Make sure you include stride in your calculation\033[90m\n")

    if test_count == 2:
        print("\033[92mFirst Test: All tests passed!")


def conv_forward_test_2(target):
    # Test 1
    np.random.seed(3)
    A_prev = np.random.randn(2, 5, 7, 4)
    W = np.random.randn(3, 3, 4, 8)
    b = np.random.randn(1, 1, 1, 8)

    Z1 = target(A_prev, W, b, {"pad": 3, "stride": 1})
    Z_shape = Z1.shape
    assert Z_shape[0] == A_prev.shape[0], f"m is wrong. Current: {Z_shape[0]}.  Expected: {A_prev.shape[0]}"
    assert Z_shape[1] == 9, f"n_H is wrong. Current: {Z_shape[1]}.  Expected: 9"
    assert Z_shape[2] == 11, f"n_W is wrong. Current: {Z_shape[2]}.  Expected: 11"
    assert Z_shape[3] == W.shape[3], f"n_C is wrong. Current: {Z_shape[3]}.  Expected: {W.shape[3]}"

    # Test 2
    Z2 = target(A_prev, W, b, {"pad": 0, "stride": 2})
    assert (Z2.shape == (2, 2, 3, 8)), "Wrong shape. Don't hard code the pad and stride values in the function"

    # Test 3
    W = np.random.randn(5, 5, 4, 8)
    b = np.random.randn(1, 1, 1, 8)
    Z3  = target(A_prev, W, b, {"pad": 6, "stride": 1})
    Z_shape = Z3.shape
    assert Z_shape[0] == A_prev.shape[0], f"m is wrong. Current: {Z_shape[0]}.  Expected: {A_prev.shape[0]}"
    assert Z_shape[1] == 13, f"n_H is wrong. Current: {Z_shape[1]}.  Expected: 13"
    assert Z_shape[2] == 15, f"n_W is wrong. Current: {Z_shape[2]}.  Expected: 15"
    assert Z_shape[3] == W.shape[3], f"n_C is wrong. Current: {Z_shape[3]}.  Expected: {W.shape[3]}"

    Z_means = np.mean(Z3)
    expected_Z = -0.5384027772160062

    expected_conv = np.array([[1.98848968, 1.19505834, -0.0952376, -0.52718778],
                              [-0.32158469, 0.15113037, -0.01862772, 0.48352879],
                              [0.76896516, 1.36624284, 1.14726479, -0.11022916],
                              [0.38825041, -0.38712718, -0.58722031, 1.91082685],
                              [-0.45984615, 1.99073781, -0.34903539, 0.25282509],
                              [1.08940955, 0.02392202, 0.39312528, -0.2413848],
                              [-0.47552486, -0.16577702, -0.64971742, 1.63138295]])

    assert np.isclose(Z_means, expected_Z), f"Wrong Z mean. Expected: {expected_Z} got: {Z_means}"
    assert np.allclose(A_prev[1, 2], expected_conv), f"Values in Z are wrong"

    print("\033[92mSecond Test: All tests passed!")


def pool_forward_test_1(target):
    # Test 1
    A_prev = np.random.randn(2, 5, 7, 3)
    A = target(A_prev, {"stride": 2, "f": 2}, mode="average")
    A_shape = A.shape
    assert A_shape[0] == A_prev.shape[0], f"Test 1 - m is wrong. Current: {A_shape[0]}.  Expected: {A_prev.shape[0]}"
    assert A_shape[1] == 2, f"Test 1 - n_H is wrong. Current: {A_shape[1]}.  Expected: 2"
    assert A_shape[2] == 3, f"Test 1 - n_W is wrong. Current: {A_shape[2]}.  Expected: 3"
    assert A_shape[3] == A_prev.shape[3], f"Test 1 - n_C is wrong. Current: {A_shape[3]}.  Expected: {A_prev.shape[3]}"

    # Test 2
    A_prev = np.random.randn(4, 5, 7, 4)
    A = target(A_prev, {"stride": 1, "f": 5}, mode="max")
    A_shape = A.shape
    assert A_shape[0] == A_prev.shape[0], f"Test 2 - m is wrong. Current: {A_shape[0]}.  Expected: {A_prev.shape[0]}"
    assert A_shape[1] == 1, f"Test 2 - n_H is wrong. Current: {A_shape[1]}.  Expected: 1"
    assert A_shape[2] == 3, f"Test 2 - n_W is wrong. Current: {A_shape[2]}.  Expected: 3"
    assert A_shape[3] == A_prev.shape[3], f"Test 2 - n_C is wrong. Current: {A_shape[3]}.  Expected: {A_prev.shape[3]}"

    # Test 3
    np.random.seed(1)
    A_prev = np.random.randn(2, 5, 5, 3)

    A = target(A_prev, {"stride": 1, "f": 2}, mode="max")

    assert np.allclose(A[1, 1], np.array([[1.19891788, 0.74055645, 0.07734007],
                                          [0.31515939, 0.84616065, 0.07734007],
                                          [0.69803203, 0.84616065, 1.2245077],
                                          [0.69803203, 1.12141771, 1.2245077]])), "Wrong value for A[1, 1]"


    A = target(A_prev, {"stride": 1, "f": 2}, mode="average")

    assert np.allclose(A[1, 1], np.array([[0.11583785, 0.34545544, -0.6561907],
                                          [-0.2334108, 0.3364666, -0.69382351],
                                          [0.25497093, -0.21741362, -0.07342615],
                                          [-0.04092568, -0.01110394, 0.12495022]])), "Wrong value for A[1, 1]"

    print("\033[92mAll tests passed!")


def pool_forward_test_2(target):
    np.random.seed(1)
    A_prev = np.random.randn(2, 5, 5, 3)

    A = target(A_prev, {"stride": 2, "f": 3}, mode="max")

    assert np.allclose(A[0], np.array([[[1.74481176, 0.90159072, 1.65980218],
                                        [1.74481176, 1.6924546, 1.65980218]],
                                       [[1.13162939, 1.51981682, 2.18557541],
                                        [1.13162939, 1.6924546,
                                         2.18557541]]])), "Wrong value for A[0] in mode max. Make sure you have included stride in your calculation"

    A = target(A_prev, {"stride": 2, "f": 3}, mode="average")

    assert np.allclose(A[1], np.array([[[-0.17313416, 0.32377198, -0.34317572],
                                        [0.02030094, 0.14141479, -0.01231585]],
                                       [[0.42944926, 0.08446996, -0.27290905],
                                        [0.15077452, 0.28911175,
                                         0.00123239]]])), "Wrong value for A[1] in mode average. Make sure you have included stride in your calculation"

    print("\033[92mAll tests passed!")