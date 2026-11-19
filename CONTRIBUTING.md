# Contributing

## Releases

### Publishing to PyPI via Docker

Run on macOS, Linux (x86 and arm) to cover all architectures:

```sh
docker run --rm -v $(pwd):/io ghcr.io/pyo3/maturin:v1.11.5 publish -u __token__ -p PYPI_TOKEN
```

### Local Maturin build

```sh
pip install ziglang pipx
pipx install maturin
maturin publish -u __token__ -p PYPI_TOKEN --target=x86_64-unknown-linux-gnu --zig
```

macOS targets can be built without `--zig`:

```sh
rustup target add x86_64-apple-darwin aarch64-apple-darwin
maturin publish -u __token__ -p PYPI_TOKEN --target=aarch64-apple-darwin
```
