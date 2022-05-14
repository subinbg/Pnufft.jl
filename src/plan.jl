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


# function makeplan(nftype::Int, iflag::Int, ::Type{R},
#     x::NTuple{D,CuArray{R,1}}, nu::NTuple{D,CuArray{R,1}}; tol::Real=1e-6) #=
#     =# where {D, R<:FLT}

#     @assert (D in (1,2,3)) "Unsupported dimension: $D"
#     @assert (nftype == 3) "Unknown nufft type: $nftype"
#     @assert (iflag in (1,-1)) "Unknown iflag: $iflag"

#     Nnupts_x = length(x[1])
#     Nnupts_nu = length(nu[1])
#     for d = 2:D
#         @assert Nnupts_x == length(x[d]) "Mismatch in the number of source pts"
#         @assert Nnupts_nu == length(nu[d]) "Mismatch in the number of target pts"
#     end
    
#     X1 = X2 = X3 = one(R)
#     Nu1 = Nu2 = Nu3 = one(R)
#     X1 = maxabs(x[1])
#     Nu1 = max(maxabs(nu[1]), 1/X1) # maybe 1/X1/(2π)
#     if D > 1
#         X2 = maxabs(x[2])
#         Nu2 = max(maxabs(nu[2]), 1/X2)
#     end
#     if D > 2
#         X3 = maxabs(x[3])
#         Nu3 = max(maxabs(nu[3]), 1/X3)
#     end

#     w::R = ceil(Int, -log10(tol/10.0))
#     beta::R = 1.5π * w # 2.3

#     Nup1 = Nup2 = Nup3 = 1
#     if D == 1
#         Nup1 = next235even(ceil(Int, 8*X1*Nu1 + w))
#         Nup = (Nup1,)
#     elseif D == 2
#         Nup1 = next235even(ceil(Int, 8*X1*Nu1 + w))
#         Nup2 = next235even(ceil(Int, 8*X2*Nu2 + w))
#         Nup = (Nup1, Nup2)
#     else
#         Nup1 = next235even(ceil(Int, 8*X1*Nu1 + w))
#         Nup2 = next235even(ceil(Int, 8*X2*Nu2 + w))
#         Nup3 = next235even(ceil(Int, 8*X3*Nu3 + w))
#         Nup = (Nup1, Nup2, Nup3)
#     end
#     Nupup = 2 .* Nup # we need not call next235even here

#     gamma1 = gamma2 = gamma3 = one(R)
#     if D == 1
#         gamma1 = Nup1 / (4*Nu1)
#         gamma = (gamma1,)
#     elseif D == 2
#         gamma1 = Nup1 / (4*Nu1)
#         gamma2 = Nup2 / (4*Nu2)
#         gamma = (gamma1, gamma2)
#     else
#         gamma1 = Nup1 / (4*Nu1)
#         gamma2 = Nup2 / (4*Nu2)
#         gamma3 = Nup3 / (4*Nu3)
#         gamma = (gamma1, gamma2, gamma3)
#     end

#     # Here, we use Nup for the argument "N" in makeplan for type 1.
#     # It does not have any effect, since we use plan1 only for interpolation.
#     # Dewindow is done in plan2 & execute! for type 3
#     plan1 = makeplan(1, iflag, R, Nup, Nup, Nnupts_x, tol=tol)
#     plan2 = makeplan(2, iflag, R, Nup, Nupup, Nnupts_nu, tol=tol)

#     # This is a bit dirty, but...
#     nu_cpu = Array{R}(undef, Nnupts_nu)
#     nuFkernel_cpu = Array{R}(undef, Nnupts_nu)
#     for d = 1:D
#         copyto!(nu_cpu, nu[d])
#         if d == 1
#             nuFkernel_cpu .= make_Fkernel.(nu_cpu, Nup[d], beta, w, gamma[d])
#         else
#             nuFkernel_cpu .*= make_Fkernel.(nu_cpu, Nup[d], beta, w, gamma[d])
#         end
#     end
#     nuFkernel = CuArray(nuFkernel_cpu)

#     C = complex(R)
#     FFTtype = typeof(plan1.inplaceFFT)
#     Plan3{D,R,C,FFTtype}(plan1, plan2, nuFkernel, gamma1, gamma2, gamma3)
# end