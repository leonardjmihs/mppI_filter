from setuptools import setup, find_packages
import os

# with open("README.md", "r") as fh:
#     long_description = fh.read()

# with open("requirements.txt", "r") as f:
#     requirements = f.read().splitlines()

package_name = 'mppi_eece'
packages=[package_name, 
        *(os.path.join(package_name,pkg) for pkg in find_packages(package_name, exclude=['data','training','experiments', 'controllers'])), 
        ]
print(packages)    
setup(
    name=package_name,
    version='0.1.0',
    author='Your Name',
    author_email='your.email@example.com',
    description='MPPI implementation in JAX.',
    url='httpshttps://github.com/your_username/mppi_eece',
    packages=packages,
)