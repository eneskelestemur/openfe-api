# Contributing

## Setting up

```bash
conda env create -n openfe-api -f container/environment.yml
conda activate openfe-api
pip install -e '.[service,dev,docs]'
```

## Checks

```bash
pytest                 # fast tests, offline, no GPU
pytest -m slow         # adds the tests that build real transformations
python -m pyright      # must report zero errors
ruff format            # the one mechanical style
ruff check             # must report no findings
mkdocs serve           # docs at http://127.0.0.1:8000
```

Formatting and linting are not a matter of taste here: the rule set is pinned in
`pyproject.toml` so a tool upgrade cannot move it. Run `ruff format` and `ruff check` before
calling a change finished.

## House rules

The ones that shape most changes:

- Every public function, method and class gets a **Google-style docstring** with full
  `Args` / `Returns` / `Raises` sections.
- Tests are written **with** the implementation, never deferred. Every input-validation rule
  gets a negative test proving it fails with a clear message. `tests/` mirrors the package, so
  a change to `prep/series.py` belongs with `tests/prep/test_series.py`.
- **Fail loudly on bad input.** Messages name the file, the chain or residue, the rule that
  was broken, and how to fix it.
- **Never silently change chemistry.** Anything that alters a molecule is opt-in, off by
  default, and warns when used.
- **One mechanical style**, applied by `ruff format` rather than agreed by hand. American
  English in code, docstrings and documentation.
- Every module declares `__all__`. A subpackage's `__init__.py` holds only its docstring, so
  there is no second export list to keep in sync.
- No new runtime dependency without discussion.
- No module may be named after a standard library module; a test enforces this.

## Releasing

The package version lives in `pyproject.toml`; `openfe_api.__version__` reads it back from the
installed package metadata, and `openfe-api version` prints it. `CITATION.cff` carries its own
copy for the citation, and a test fails if the two drift. Cutting a release is:

1. Bump `version` in `pyproject.toml`.
2. Bump `version` and `date-released` in `CITATION.cff`.
3. Move `CHANGELOG.md`'s unreleased entries under the new version with today's date.
4. Tag it: `git tag -a v0.2.0 -m "v0.2.0" && git push --tags`.

[Semantic versioning](https://semver.org/spec/v2.0.0.html) applies to the request schema, the
CLI and the public library API. The campaign directory layout is part of that contract too: a
change that makes an existing campaign directory unreadable is a major change.

## Continuous integration

`.github/workflows/ci.yml` runs the same checks listed above on every push and pull request,
in a conda environment built from `container/environment.yml`. A push to `main` that passes
also publishes the documentation to GitHub Pages.
`.github/workflows/container.yml` builds the container image when its recipe or the package
changes, to prove it still builds; it publishes no image.

The Pages deploy needs one setting on the repository, once: **Settings → Pages → Build and
deployment → Source: GitHub Actions**. Until that is set the deploy job fails while the checks
still pass.

## Layout

```text
src/openfe_api/
├── schema/      request and profile models
├── prep/        structures in, simulation-ready files out
├── protocols/   settings, charges, transformations
├── execution/   Slurm and local backends
├── results/     gathering and quality checks
├── campaign.py  the state machine
├── graph.py     network connectivity
├── cli.py       Typer front end
└── service.py   FastAPI front end
```

`tests/` mirrors this, one directory per subpackage, with shared builders in
`tests/helpers.py` and the cross-cutting tests at the top. A change to `prep/series.py`
belongs with `tests/prep/test_series.py`. Same-named test files in different directories need
`--import-mode=importlib`, and the builders need `tests/` on the path; both are set in
`pyproject.toml`.

The CLI and the service are thin layers: no logic lives in either.
