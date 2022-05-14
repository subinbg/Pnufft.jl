# Pnufft

[![Build Status](https://github.com/subinbg/Pnufft.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/subinbg/Pnufft.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![codecov](https://codecov.io/gh/subinbg/Pnufft.jl/branch/main/graph/badge.svg?token=w5x8FIr3JD)](https://codecov.io/gh/subinbg/Pnufft.jl)

A Julia package for non-uniform fast Fourier transform (NUFFT).      
- This pacakge supports both CPU (via `FLoops.jl`) and GPU (via `CUDA.jl`)    
- The NUFFT type 1-3 operations on dimension 1-3
- Both `+1` and `-1` (sign of the exponent) are supported

A typical example usage is (check `runtests.jl` for more detailed examples):     
```julia
plan = plan_nufft!(X, N)
plan(target, iflag, source)
```