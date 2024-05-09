import setuptools

install_requires = [
    'z3-solver',  # tested version: 4.13.0
    'more_itertools',
    'matplotlib'
]

setuptools.setup(
  name=             'tessel',
  version=          '0.3',
  author=           'Zhiqi Lin',
  author_email=     'zhiqi.0@outlook.com',
  description=      'Schedule plan searching for composing micro-batch executions',
  long_description= '',
  packages=         ['tessel'],
  python_requires=  '>=3.8',
  install_requires= install_requires,
)
