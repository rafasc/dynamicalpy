from setuptools import setup


setup(
    name='dynamicalpy',
    version='0.0.0',
    author='Rafael Ascensão',
    author_email='rafa.almas@gmail.com',
    install_requires=[
        "ipython",
        "matplotlib",
        "networkx",
        "numpy",
        "pygraphviz",
        "scipy"],
    extras_require = {
        'Qt5': ["PyQt5"]
    },
    packages=['dynamicalpy'],
)


