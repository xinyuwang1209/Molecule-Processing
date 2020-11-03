from setuptools import setup, find_packages


setup(name='moleculeprocessing',
      version=0.01,
      python_requires='>=3.5.0',
      packages=find_packages(include=['MoleculeProcessing','MoleculeProcessing.*']),
      classifiers=[
			'Environment :: Console',
			'Intended Audience :: Developers',
			'Operating System :: OS Independent',
			'Programming Language :: Python',
			'Programming Language :: Python :: 3'
		],
      description=('MoleculeProcessing'),
      author='',
      author_email='',
      package_data={
          '': ['*.csv', '*.h5', '*.gz','*.pt'],
      },
      platforms = 'any',
      )
