from setuptools import setup, find_packages

setup(
    name="edm_diffusion_policy",
    version="0.1",
    description="Minimal implementation of a EDM-based Diffusion Policy for Robotics.",
    license="MIT",
    author="Moritz Reuss",
    url="https://github.com/mbreuss/edm_diffusion_policy",
    packages=find_packages(),
    install_requires=[
        "torch",
        "torchsde",
        "torchdiffeq",
        "einops",
        "numpy",
        "omegaconf"
    ]
)