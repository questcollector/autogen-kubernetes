## Contribute

This project's structure conforms microsoft/autogen python package

```
├── LICENSE
└── python
    ├── run_task_in_pkgs_if_exist.py
    ├── pyproject.toml
    ├── shared_tasks.toml
    ├── uv.lock
    └── packages
        └── autogen-kubernetes
            ├── tests
            │   ├── test_utils.py
            │   ├── conftest.py
            │   ├── test_kubernetees_code_executor.py
            │   ├── test-pod.yaml
            │   └── test-volume.yaml
            ├── LICENSE-CODE
            ├── pyproject.toml
            ├── README.md
            └── src
                └── autogen_kubernetes
                    ├── py.typed
                    └── code_executors
                        ├── _utils.py
                        └── _kubernetes_code_executor.py
```

### Install uv

Install uv according to your environment.

https://docs.astral.sh/uv/getting-started/installation/

### Sync project

```sh
cd autogen-kubernetes/python
uv venv --python python >=3.10
source .venv/bin/activate
uv sync --all-extras
```

### Common tasks

- Format: `poe format`
- Lint: `poe lint`
- Test: `poe test`
- Mypy: `poe mypy`
- Check all: `poe check`

## Licensing

This project is licensed under the MIT License.

### Code Modification

This project includes code from the microsoft/autogen project (licensed under the MIT License), 

with modifications made by kiyoung you(questcollector), See the LICENSE-CODE file for details

### Third-Party Dependencies

This project uses the following third-party dependencies:

1. **kubernetes**
    License: Apache License, Version 2.0
    Source: https://github.com/kubernetes-client/python
2. **httpx**
    License: BSD 3-Clause "New" or "Revised"
    Source: https://github.com/encode/httpx
3. **websockets**
    License: BSD 3-Clause "New" or "Revised"
    Source: https://github.com/python-websockets/websockets
4. **PyYAML**
    License: MIT License
    Source: https://github.com/yaml/pyyaml

For details, see the LICENSE-CODE file.