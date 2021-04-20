from setuptools import setup, find_packages

def get_description():
    return "ML pipelining and benchmarking suite geared towards the physical sciences"

def get_scripts():
    return [
        "./bin/bbatch",
        "./bin/binfo",
        "./bin/binput",
        "./bin/bmark",
        "./bin/bml",
        "./bin/bmeta"
    ]

if __name__ == "__main__":
    setup(
        name="BenchML",
        version="0.1.1",
        url="https://github.com/capoe/benchml",
        description="Machine-learning and pipelining suite",
        long_description=get_description(),
        packages=find_packages(),
        scripts=get_scripts(),
        setup_requires=[],
        install_requires=["numpy", "scipy", "scikit-learn"],
        include_package_data=True,
        ext_modules=[],
        license="Apache License 2.0",
        classifiers=[
            "Development Status :: 3 - Alpha",
            "Intended Audience :: Science/Research",
            "Topic :: Scientific/Engineering :: Physics",
            "Topic :: Scientific/Engineering :: Chemistry",
            "Topic :: Scientific/Engineering :: Artificial Intelligence"
        ],
        keywords="chemical physicochemical atomistic machine learning pipelining benchmarking",
        python_requires=">=3.7",
    )

