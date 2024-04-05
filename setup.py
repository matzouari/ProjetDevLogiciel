from setuptools import setup, find_packages
setup(
name='CriminAI',
version='0.1.2',
author='ZOUARI Matis, PEREZ Lisa, SUTTER Clemence, ZHONG Zhihan',
author_email='matis.zouari@insa-lyon.fr, lisa.perez@insa-lyon.fr, clemence.sutter@insa-lyon.fr, zhihan.zhong@insa-lyon.fr',
url='https://github.com/matzouari/ProjetDevLogiciel/',
description="CriminAI est un logiciel de génération de portraits robots par IA",
packages=find_packages(),
install_requires=[
          'numpy',
          'matplotlib',
          'torch',
          'torchvision',
          'tk',
          'PIL',
      ],
classifiers=[
'Programming Language :: Python :: 3',
'License :: OSI Approved :: MIT License',
'Operating System :: OS Independent',
],
python_requires='>=3.6',
)