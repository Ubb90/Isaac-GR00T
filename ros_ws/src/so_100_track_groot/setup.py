from setuptools import setup
import sys

package_name = 'so_100_track_groot'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=[
        'setuptools',
        'numpy',
        'opencv-python',
        'draccus',
    ],
    zip_safe=True,
    maintainer='baxter',
    maintainer_email='2330834a@student.gla.ac.uk',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            f"eval_lerobot = {package_name}.eval_lerobot_ros2:main"
        ],
    },
    options={
        'build_scripts': {
            'executable': sys.executable,
        },
    },
)
