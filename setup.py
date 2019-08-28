from setuptools import setup, find_packages

setup(name='trainer',
      version='0.1',
      packages=find_packages(),
      description='CS-Project',
      author='Giuseppe Sarno',
      author_email='tomloss22@gmail.com',
      license='MIT',
      install_requires=[
          'keras',
          'h5py', 'GoogleAppEngineCloudStorageClient', 'google', 'tensorflow'
      ],
      zip_safe=False)
