from setuptools import setup, find_packages

setup(
    name="PyCoShREM",
    version="0.0.2",
    description="Complex shearlet based edge and ridge measurement",
    author="RG Computational Data Analysis, University Bremen: Rafael Reisenhofer, Jonas Wloka",
    author_email='reisenhofer@math.uni-bremen.de',
    url='http://www.math.uni-bremen.de/cda/software.html',
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "Programming Language :: Python :: 3.5"
    ],
    keywords='image-processing shearlet edge ridge',
    packages=find_packages(exclude=[
        'docs',
        'tests'
    ]),
    install_requires=[
        'numpy',
        'scipy',
        'pillow',
        'matplotlib',
        'scikit-image'
    ],
    download_url='https://github.com/rgcda/pycoshrem/archive/0.0.2.tar.gz'
)
