from setuptools import setup, find_packages

def readme():
    try:
        with open('README.md', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        return ''

setup(
    name='l0_amplitude_and_phase',
    version='0.1.0',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=['numpy>=1.21.0'],
    python_requires='>=3.7',
    author='Abdulhameed Yunusa',
    author_email='ayunusa@indracompany.com',
    description='A module for generating amplitude and phase from complex IQ components in Level-0 SAR data',
    long_description=readme(),
    long_description_content_type='text/markdown',
    license='MIT',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Intended Audience :: Developers',
        'Topic :: SAR :: Scientific/Engineering :: Information Analysis',
    ],
)