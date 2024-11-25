# pip install setuptools
from setuptools import setup

setup(
    name='MARSS112524',
    # version='0.1.0',    
    # description='An example Python package',
    # url='https://github.com/shuds13/pyexample',
    # author='Stephen Hudson',
    # author_email='shudson@anl.gov',
    # license='BSD 2-clause',
    packages=['MARSS112524'],
    install_requires=[
        'mpi4py>=2.0',
        'numpy',
        'nibabel',
        'shutil', 
        'pandas',
        'matplotlib',
        'scipy',
        'networkx',
        'traits',
        'python-dateutil',
        'Sphinx',
        'traitsui',
        'nipype[all]',
        'seaborn',
    ],

    # classifiers=[
    #     'Development Status :: 1 - Planning',
    #     'Intended Audience :: Science/Research',
    #     'License :: OSI Approved :: BSD License',  
    #     'Operating System :: POSIX :: Linux',        
    #     'Programming Language :: Python :: 2',
    #     'Programming Language :: Python :: 2.7',
    #     'Programming Language :: Python :: 3',
    #     'Programming Language :: Python :: 3.4',
    #     'Programming Language :: Python :: 3.5',
    # ],
)
