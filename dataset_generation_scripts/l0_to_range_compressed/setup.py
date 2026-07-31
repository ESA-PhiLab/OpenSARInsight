from setuptools import setup, find_packages

setup(
    name='l0_to_range_compressed',
    version='0.1',
    description='A Python pipeline for processing Sentinel-1 Level-0 SAR data through range compression and rescaling.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='Abdulhameed Yunusa',
    author_email='ayunusa@indracompany.com',
    url='',
    packages=find_packages(),
    python_requires='>=3.12.3',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    include_package_data=True,
)