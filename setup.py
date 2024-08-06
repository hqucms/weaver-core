import os
from setuptools import setup, find_packages

with open('README.md') as f:
    long_desc = f.read()

requirements_path = os.path.join(os.path.dirname(__file__), 'requirements.txt')
install_requires = []
with open(requirements_path) as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        if line.startswith('#'):
            continue
        install_requires.append(line)

setup(name="weaver-core",
      version='0.4.17',
      description="A streamlined deep-learning framework for high energy physics",
      long_description_content_type="text/markdown",
      author="H. Qu, C. Li",
      url="https://github.com/hqucms/weaver-core",
      long_description=long_desc,
      entry_points={'console_scripts':
                    ['weaver = weaver.train:main']},
      packages=find_packages(),
      zip_safe=False,
      classifiers=[
          "Operating System :: OS Independent",
          'Programming Language :: Python',
          'Programming Language :: Python :: 3',
          'Programming Language :: Python :: 3.7',
          'Topic :: Scientific/Engineering',
          'Topic :: Scientific/Engineering :: Physics',
      ],
      python_requires='>=3.7',
      install_requires=install_requires,
      )
