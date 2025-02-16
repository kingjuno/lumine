import os
import subprocess
from setuptools import setup
from setuptools.command.install import install

class CMakeBuild(install):
    def run(self):
        build_dir = os.path.abspath("build")
        os.makedirs(build_dir, exist_ok=True)

        # Run CMake EXACTLY as you do manually
        subprocess.check_call(["cmake", "-S", "cmake", "-B", build_dir])
        subprocess.check_call(["cmake", "--build", build_dir])

        # Ensure tensor.so exists
        so_path = os.path.join(build_dir, "tensor.so")
        if not os.path.exists(so_path):
            raise RuntimeError("ERROR: tensor.so was NOT built! Check CMake output.")

        # Proceed with installation
        install.run(self)

setup(
    name="lumine",
    version="0.1",
    packages=["lumine"],
    include_package_data=True,
    cmdclass={"install": CMakeBuild},
)
