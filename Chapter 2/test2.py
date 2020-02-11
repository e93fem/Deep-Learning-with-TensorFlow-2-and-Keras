import tensorflow as tf

def linear_layer(x):
    return 3 * x + 2

@tf.function
def simple_nn(x):
    return tf.nn.relu(linear_layer(x))

def simple_function(x):
    return 3 * x

print(simple_nn)
print(simple_function)

# internal look at the auto-generated code
print(tf.autograph.to_code(simple_nn.python_function, experimental_optional_features=None))

