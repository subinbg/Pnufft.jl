# Pnufft

[![Julia version](https://img.shields.io/badge/Julia-1.8-informational?logo=julia&logoColor=white&style=flat)](https://julialang.org/)
[![MIT license](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/subinbg/Pnufft.jl/blob/main/LICENSE)
[![Build Status](https://github.com/subinbg/Pnufft.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/subinbg/Pnufft.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/subinbg/Pnufft.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/subinbg/Pnufft.jl)

Non-uniform fast Fourier transform (NUFFT), purely written in Julia. This package is based on [1-2].

## Mathematical definitions

- The type-1 transform (non-uniform to uniform) computes Fourier components on uniform grid points $\mathbf{q} \in \mathcal{I}\_{N\_{1}} \times \cdots \times \mathcal{I}\_{N\_{D}}$ from coefficients on non-uniform grid points $\mathbf{x}\_{m} \in [-1/2,1/2]^{D}$:

$$ \tilde{f}\_{\mathbf{q}} = \sum\_{m=1}^{M} c\_{m} e^{\pm 2\pi i \left\langle \mathbf{q} \mid \mathbf{x}\_m \right\rangle } $$

where $\mathcal{I}\_{N_{i}} = \left\\{-N\_{i}/2, -N\_{i}/2+1, \cdots N\_{i}/2-1 \right\\}$.

- The type-2 transform (uniform to non-uniform) computes Fourier components on non-uniform grid points $\mathbf{x}\_{m} \in [-1/2,1/2]^{D}$ from coefficients on uniform grid points $\mathbf{q} \in \mathcal{I}\_{N\_{1}} \times \cdots \times \mathcal{I}\_{N_{D}}$:

$$ c\_{m} = \sum_{\mathbf{q} \in \mathcal{I}^{D}\_{N}} \tilde{f}\_{\mathbf{q}} e^{\pm 2\pi i \left\langle \mathbf{x}\_{m} \mid \mathbf{q} \right\rangle} $$

where $\mathcal{I}^{D}\_{N} = \mathcal{I}\_{N\_{1}} \times \cdots \times \mathcal{I}\_{N\_{D}}$.

- The type-3 transform (non-uniform to non-uniform) computes Fourier components on non-uniform grid points $\nu\_{p}\in\mathbb{R}^{D}$ from coefficients on non-uniform grid points $\mathbf{x}\_{m} \in [-1/2,1/2]^{D}$:

$$ \tilde{f}\_{p} = \sum\_{m=1}^{M} c\_{m} e^{\pm 2\pi i \left\langle \nu\_{p} \mid \mathbf{x}\_{m} \right\rangle} $$

## Supported operations

- This pacakge supports both CPU (via `FLoops.jl`) and GPU (via `CUDA.jl`)
- The NUFFT type 1-3 operations on dimension $D=1,2,3$
- Both signs `+1` and `-1` in the exponents, $\exp \left( \pm 2\pi i \cdots \right)$, are supported (you can choose the sign with the `iflag` argument; see below).

## How to use

A typical example usage is (check `runtests.jl` for more detailed examples):

```julia
# X and N specify non-uniform and uniform grid points
plan = plan_nufft!(X, N) 
# source refers to cofficients that are Fourier transformed
# target refers to a storage that Fourier components are stored
plan(target, iflag, source)
```

## References

[1] A. H. Barnett et al., A parallel nonuniform fast Fourier transform library based on an “exponential of semicircle" kernel, SIAM Journal on Scientific Computing, 41:5.

[2] D. Potts et al., Uniform error estimates for nonequispaced fast Fourier transforms, Sampling Theory, Signal Processing, and Data Analysis, 19:17.
