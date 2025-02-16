import os
import subprocess
import sys
from setuptools import setup
from setuptools.command.build_ext import build_ext

class CMakeBuild(build_ext):
    def run(self):
        build_dir = os.path.abspath(self.build_temp)
        os.makedirs(build_dir, exist_ok=True)

        cmake_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "cmake"))

        subprocess.check_call(["cmake", cmake_dir], cwd=build_dir)
        subprocess.check_call(["cmake", "--build", "."], cwd=build_dir)

setup(
    name="ctensor",
    version="0.1",
    ext_modules=[],
    cmdclass={"build_ext": CMakeBuild},
)
