from setuptools import setup

install_requires = [
    'MDAnalysis',
    'jax>=0.2.12',
    'jaxlib>=0.1.65',
    'jax-md==0.1.13',
    'optax>=0.0.6',
    'dm-haiku>=0.0.4',
    'sympy>=1.8'
]

setup(
        name='DiffTRe',
        version='0.1',
        license='Apache 2.0',
        description='Differentiable Trajectory Reweighting',
        author='Stephan Thaler',
        author_email='stephan.thaler@tum.de',
        packages=['DiffTRe'],
        install_requires=install_requires,
        zip_safe=False,
)