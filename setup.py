from setuptools import setup
setup(
    name = 'auth-pipeline',
    version = '0.1.0',
    packages = ['authpipe'],
    entry_points = {
        'console_scripts': [
            'authpipe = authpipe.__main__:main'
        ]
    })