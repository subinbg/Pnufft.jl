struct NuFFTPlan{T,D,FK,B,X,UP,FT,IT}

    # Window function
    m::Int
    beta::T
    Fkernel::NTuple{D,FK}

    # Sorting
    binSize::NTuple{D,Int}
    Nbin::NTuple{D,Int}
    binCount::B
    binStartIdx::B
    posIdxinBin::B
    sortedIdx::B

    # number of points
    N::NTuple{D,Int}
    Nup::NTuple{D,Int}
    Nnupts::Int

    # non-uniform grid
    x::X

    # Fourier transform
    upgrid::Base.RefValue{UP} # largest allocation, so provide `free!` function...
    FFT::FT
    IFFT::IT
end

struct NuFFTPlan3{T,D,FK,B,X,UP,FT,IT}
    plan1::NuFFTPlan{T,D,FK,B,X,UP,FT,IT}
    plan2::NuFFTPlan{T,D,FK,B,X,UP,FT,IT}
    nuFkernel::FK
end

free!(plan::NuFFTPlan{T,D}) where {T,D} = (plan.upgrid[] = typeof(plan.upgrid[])(undef, [0 for idx in 1:D]...))
function free!(plan::NuFFTPlan3)
    free!(plan.plan1)
    free!(plan.plan2)
end



"""
plan function for type 1,2 NUFFT operations
"""
function plan_nufft!(X::AbstractArray{T,2}, N::NTuple{D,Int};
    gpu::Bool=false, reltol=1e-6, of=2.0, binSize::NTuple{D,Int}=defaultBinSize[D]) where {T,D}

    Nup = ceil.(Int, N .* of)
    Nup = Nup .+ Nup .% 2 # make even numbers
    Nbin = ceil.(Int, Nup ./ binSize)

    if gpu && !CUDA.functional()
        @warn "gpu=true, but CUDA not detected... automatically set gpu=false"
        gpu = false
    end

    X_device = to_device(X, gpu)
    plan_nufft!(X_device, N, of, Nup, Nbin, binSize, reltol)
end

function plan_nufft!(X::AbstractArray{T,2}, 
    N::NTuple{D,Int}, of, Nup::NTuple{D,Int}, 
    Nbin::NTuple{D,Int}, binSize::NTuple{D,Int}, 
    reltol) where {T,D}

    check_args(D, X, N, of ,Nbin)

    w = ceil(Int, -log10(reltol/10.0))
    m = ceil(Int, w / 2.0) # iseven(w) ? w ÷ 2 : w ÷ 2 + 1
    beta::T = 2π * m * (1 - 1/(2*of))

    Nnupts = size(X, 1)
    binCount = similar(X, Int, *(Nbin...))
    binStartIdx = similar(X, Int, *(Nbin...))
    posIdxinBin = similar(X, Int, Nnupts)
    sortedIdx   = similar(X, Int, Nnupts)

    Fkernel = []
    for d in 1:D
        temp = similar(X, T, N[d])
        copyto!(temp, make_Fkernel(N[d], Nup[d], beta, m))
        push!(Fkernel, temp)
    end
    Fkernel_tuple = Tuple(Fkernel)

    upgrid = similar(X, Complex{T}, Nup)
    upref = Ref(upgrid)
    FFT = plan_fft!(upgrid)
    IFFT = plan_bfft!(upgrid)
    
    plan = NuFFTPlan(m, beta, Fkernel_tuple, 
        binSize, Nbin, binCount, binStartIdx, posIdxinBin, sortedIdx, 
        N, Nup, Nnupts, X, upref, FFT, IFFT)

    BinSort!(plan)
    return plan 
end


"""
plan function for type 3 NUFFT operation
"""
function plan_nufft!(X::AbstractArray{T,2}, S::AbstractArray{T,2};
    gpu::Bool=false, reltol=1e-6, of=2.0, binSize::NTuple{D,Int}=defaultBinSize[size(X,2)]) where {T,D}

    LX = maximum(X, dims=1)
    LS = maximum(S, dims=1)
    MX = minimum(X, dims=1)
    MS = minimum(S, dims=1)

    ALX = Array(max.(abs.(LX), abs.(MX)))[1:D]
    ALS = Array(max.(abs.(LS), abs.(MS)))[1:D]
    NU = max.(ALS, one(T)./ALX) # maybe 1/ALX/(2π)

    w = ceil(Int, -log10(reltol/10.0))
    m = ceil(Int, w / 2.0)
    beta::T = 2π * m * (1 - 1/(2*of))

    Nup = ceil.(Int, 4 * of .* ALX .* NU .+ w)
    Nup = Nup .+ Nup .% 2 # make even numbers
    Nupup = ceil.(Int, of .* Nup)
    Nupup = Nupup .+ Nupup .% 2 # make even numbers

    gamma = Nup ./ (2 .* convert(T, of) .* NU)
    gamma_device = to_device(reshape(gamma, 1, length(gamma)), gpu)
    Nup_device = to_device(reshape(Nup, 1, length(Nup)), gpu)

    
    if gpu && !CUDA.functional()
        @warn "gpu=true, but CUDA not detected... automatically set gpu=false"
        gpu = false
    end
    X_device = to_device(X, gpu)
    S_device = to_device(S, gpu)

    # For S, we additionally divide gamma1~3 by plan1.Nup1~3
    # to follow the definition of type 2 transform.
    X_device ./= gamma_device
    S_device .*= gamma_device ./ Nup_device
    
    # Here, we use Nup for the argument "N" in makeplan for type 1.
    # It does not have any effect, since we use plan1 only for interpolation.
    # Dewindow is done in plan2 & execute! for type 3
    Nup_t = Tuple(Nup)
    Nupup_t = Tuple(Nupup)
    plan1 = plan_nufft!(X_device, Nup_t, of, Nup_t, ceil.(Int, Nup_t ./ binSize), binSize, reltol)
    plan2 = plan_nufft!(S_device, Nup_t, of, Nupup_t, ceil.(Int, Nupup_t ./ binSize), binSize, reltol)

    # This is a bit dirty, but...
    nu = Array(S)
    Fkernel = make_Fkernel.(nu[:,1], Nup[1], beta, m, gamma[1])
    for d = 2:D
        Fkernel .*= make_Fkernel.(nu[:,d], Nup[d], beta, m, gamma[d])
    end
    Fkernel_device = to_device(Fkernel, gpu)

    NuFFTPlan3(plan1, plan2, Fkernel_device)
end