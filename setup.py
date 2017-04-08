from setuptools import setup, find_packages

setup(
    name='indi',
    version='0.01',
    description='Easy to use ML library',
    url='https://github.com/upul/indi',
    author='Upul Bandara',
    author_email='upulbandara@gmail.com',
    license='MIT',
    packages=find_packages(exclude=['indi.test'])
)
