import sys

from codes.runner import Runner

sys.dont_write_bytecode = True

runner = Runner(logging=True)
runner.run()
