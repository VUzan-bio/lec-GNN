from setuptools import find_packages, setup

setup(
    name="lec-trafficking-gnn",
    version="0.1.0",
    description="LEC trafficking GNN and GRN analysis pipeline",
    package_dir={"": "src"},
    packages=find_packages("src"),
    python_requires=">=3.10",
)
