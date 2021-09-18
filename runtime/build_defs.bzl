
def if_tf_serving(if_true, if_false = []):
    return if_true if native.package_name().startswith("tensorflow_serving/") else if_false
