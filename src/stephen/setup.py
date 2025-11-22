from setuptools import setup

package_name = 'stephen'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='stephen',
    maintainer_email='stephen@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'gap_follow=stephen.gap_follow:main',
            'wall_follow=stephen.wall_follow:main',
            'scan_visual=stephen.scan_visual:main',
            'path_follow=stephen.path_follow:main',
            'test=stephen.testing:main'
        ],
    },
)
