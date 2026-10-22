"""Load the benchmark ABC into our own benchmark with a the pass/fail criteria and score on minimum parameters"""


# we need to pick a minimum validation threshold, but i think this is tricky given just the model to avoid overfitting
# So this reframes to a compression challenge where the model should be able to classify all samples across the whole mnist dataset correctly
# scores = {% of correct samples, network_size = number of parameters in the model} ranked by % first then network size