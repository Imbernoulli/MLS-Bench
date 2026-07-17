from setuptools import find_packages, setup


def get_requires():
    with open("requirements.txt", encoding="utf-8") as f:
        file_content = f.read()
        lines = [line.strip() for line in file_content.strip().split("\n") if not line.startswith("#")]
        return lines


setup(
    name="mathruler",
    version="0.1.0",
    description="A light-weight tool for evaluating LLMs in rule-based ways.",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    packages=find_packages(),
    install_requires=get_requires(),
    entry_points={"console_scripts": ["mathruler = mathruler.interface:main"]},
)
