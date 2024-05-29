# dasrust
A Python library containing selected signal processing functions written in Rust for performance.

## Requrements
This Python package requires Rust and maturin to be preinstalled.

Install rust
´´´
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
´´´

Install maturin
´´´
pip install maturin
´´´

## Installation
the library needs to be build in release mode if performace is desired.
´´´
MACOSX_DEPLOYMENT_TARGET=14.2 maturin develop --release
´´´
