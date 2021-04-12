import os
from setuptools import setup, find_packages


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    name="Projet de session IFT725",
    version="0.0.1",
    author="D'Jeff, Mohamed, et Gabriel",
    author_email="",
    description="Projet de session dans le cadre du cours IFT725 à l'Université de Sherbrooke"
                "Il porte sur l'apprentissage actif",
    license="BSD",
    keywords="Pytorch deep learning cnn active learning",
    url="http://info.usherbrooke.ca/pmjodoin/cours/ift725/index.html",
    packages=find_packages(exclude=['contrib', 'doc', 'unit_tests']),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Topic :: Utilities",
        "License :: OSI Approved :: BSD License",
    ],
)
