from setuptools import setup, find_packages

def get_description():
    return "Benchmarking and pipelining for chemical machine learning"

def get_scripts():
    return [
        "./bin/bbatch",
        "./bin/binfo",
        "./bin/bmark",
        "./bin/bxyz",
        "./bin/bml",
        "./bin/bmeta"
    ]

if __name__ == "__main__":
    setup(
        name="BenchML",
        version="0.1.0",
        url="https://github.com/capoe/benchml",
        description="Chemical ML workbench",
        long_description=get_description(),
        packages=find_packages(),
        scripts=get_scripts(),
        setup_requires=[],
        install_requires=["numpy", "scipy", "scikit-learn"],
        include_package_data=True,
        ext_modules=[],
        license="Apache License 2.0",
        classifiers=[
            # TODO
        ],
        keywords="chemical machine learning pipelining benchmarking",
        python_requires=">=3.7",
    )

