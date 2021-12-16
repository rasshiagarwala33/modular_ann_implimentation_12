import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

PROJECT_NAME="modular_ann_implimentation_12"
USER_NAME="rasshiagarwala33"    

setuptools.setup(
    name="src",
    version="0.0.2",
    author=USER_NAME,
    author_email="agarwalrashi543212gmail.com",
    description="Its an modular implemetation of multilevel perceptron",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url=f"https://github.com/{USER_NAME}/{PROJECT_NAME}",
    project_urls={
        "Bug Tracker": f"https://github.com/{USER_NAME}/{PROJECT_NAME}/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages=["src"],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.7",
    install_requires=["tensorflow","numpy","matplotlib","seaborn","pandas"]
)