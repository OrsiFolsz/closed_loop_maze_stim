from setuptools import setup, find_packages

setup(
    name='closed_loop_maze_stim',  # Replace with your project name
    version='0.1',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    include_package_data=True,
    install_requires=[
        'numpy',
        'networkx',
        'pandas',
        'python-osc',
        'scipy',
        'json',
        'os',
        'cv2',
        'moviepy',
        'csv',
        'python-osc'
    ]
)
