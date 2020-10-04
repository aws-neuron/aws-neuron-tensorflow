import sys
import os
import subprocess
import importlib.util
import tensorflow

if tensorflow.__file__ is __file__:
    proc = subprocess.run([sys.executable, '-m', 'pip', 'install', 'tensorflow~=1.15.0', '--force', '--no-deps'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode:
        raise RuntimeError('tensorflow-neuron encountered installation problem. Please manually fix by "pip install tensorflow~=1.15.0 --force --no-deps"')
    spec = importlib.util.spec_from_file_location('tensorflow', os.path.join(os.path.dirname(__file__), 'tensorflow', '__init__.py'))
    tf = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tf)
    sys.modules['tensorflow'] = tf
