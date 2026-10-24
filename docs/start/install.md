# Install

openfe-api needs OpenFE 1.12 and its simulation stack, which come from conda-forge.

```bash
git clone https://github.com/eneskelestemur/openfe-api.git
cd openfe-api
conda env create -n openfe-api -f container/environment.yml
conda activate openfe-api
pip install -e '.[service]'          # add ,dev,docs to work on the project
openfe-api version
```

On a cluster, the [container image](../running/service.md#container-images) carries the same
environment and skips the conda solve.
