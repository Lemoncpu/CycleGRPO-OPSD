# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import re

from setuptools import find_packages, setup


def get_version() -> str:
    with open(os.path.join("verl", "__init__.py"), encoding="utf-8") as f:
        file_content = f.read()
        pattern = r"__version__\W*=\W*\"([^\"]+)\""
        (version,) = re.findall(pattern, file_content)
        return version


def get_requires(path: str) -> list[str]:
    """Read a requirements file, resolving local ``-r`` includes."""
    requirements: list[str] = []
    with open(path, encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("-r ") or line.startswith("--requirement "):
                _, included_path = line.split(maxsplit=1)
                requirements.extend(get_requires(os.path.join(os.path.dirname(path), included_path)))
            else:
                requirements.append(line)
    return requirements


extra_require = {
    "refcoco-data": get_requires("requirements/refcoco-data.txt"),
    "rl-train": get_requires("requirements/rl-train.txt"),
    "cuda-kernels": get_requires("requirements/cuda-kernels.txt"),
    "rollout-qwen3vl": get_requires("requirements/rollout-qwen3vl.txt"),
    "tracking": get_requires("requirements/tracking.txt"),
    "eval": get_requires("requirements/eval.txt"),
    "legacy": get_requires("requirements/legacy.txt"),
    "dev": ["pre-commit", "ruff"],
}


def main():
    setup(
        name="verl",
        version=get_version(),
        description="An Efficient, Scalable, Multi-Modality RL Training Framework based on veRL",
        long_description=open("README.md", encoding="utf-8").read(),
        long_description_content_type="text/markdown",
        author="verl",
        author_email="zhangchi.usc1992@bytedance.com, gmsheng@connect.hku.hk, hiyouga@buaa.edu.cn",
        license="Apache 2.0 License",
        url="https://github.com/volcengine/verl",
        package_dir={"": "."},
        packages=find_packages(where="."),
        python_requires=">=3.9.0",
        install_requires=get_requires("requirements/base.txt"),
        extras_require=extra_require,
    )


if __name__ == "__main__":
    main()
