import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

with open("LICENSE", "r") as fh:
    license = fh.read()

setuptools.setup(
    name='labvision',
    version='0.1',
    license=license,
    packages=setuptools.find_packages(
        exclude=('tests', 'docs')
    ),
    url='https://github.com/MikeSmithLabTeam/labvision',
    install_requires=[
        'opencv-python',
        'numpy',
        'matplotlib',
        'pillow',
        'scipy',
        'slicerator',
        'ffmpeg-python',
        'qtwidgets @ git+https://github.com/MikeSmithLabTeam/qtwidgets'

    ],
    test_suite='nose.collector',
    tests_require=['nose'],
    include_package_data=True,
)


