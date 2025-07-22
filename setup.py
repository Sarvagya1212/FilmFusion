from setuptools import setup, find_packages
from typing import List

Hypen_e_dot = '-e .'
def get_requirements(file_path: str) -> List[str]:
    """
    Reads the requirements file and returns a list of packages.
    """
    requirements = []
    with open(file_path, 'r') as file:
        requirements = file.readlines()
    requirements=[req.replace('\n','') for req in requirements]
    if Hypen_e_dot in requirements:
        requirements.remove(Hypen_e_dot)
    
    return requirements


setup(
    name='FilmFusion',  
    version='0.1.0',
    author='Sarvagya Jain',
    author_email='sarvagyajain5678@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)