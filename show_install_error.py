
raise ImportError(
    '\n***************************************************************\n'
    '  Sorry, tensorflow-neuron encounters an installation problem\n'
    '***************************************************************\n'
    '\n'
    '  If you are seeing this message, it indicates that your\n'
    '  tensorflow 1.15 package was accidentally uninstalled\n'
    '  during the upgrade procedure of tensorflow-neuron.\n'
    '  Please fix using the following pip install command.\n'
    '\n'
    '  pip install "tensorflow<1.16.0" --force --no-deps\n'
    '\n'
    '***************************************************************\n'
)
