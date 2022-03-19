# Use immutable struct for better performance?
# Then, I may use Base.RefValue...
struct Plan{D,R<:FLT,C<:CPX,FFT<:CUFFT.cCuFFTPlan}

    # Basic setting
    nftype::Int
    iflag::Int

    # Window function
    w::R
    beta::R
    Fkernel1::CuArray{R,1,cuDevBuffer}
    Fkernel2::CuArray{R,1,cuDevBuffer}
    Fkernel3::CuArray{R,1,cuDevBuffer}

    # Up-sampled grid
    upgrid::CuArray{C,D,cuDevBuffer}

    # Sorting
    binSize1::Int
    binSize2::Int
    binSize3::Int
    Nbin1::Int
    Nbin2::Int
    Nbin3::Int
    binCount::CuArray{Int,1,cuDevBuffer}
    binStartIdx::CuArray{Int,1,cuDevBuffer}
    posIdxinBin::CuArray{Int,1,cuDevBuffer}
    sortedIdx::CuArray{Int,1,cuDevBuffer}

    # number of points
    N1::Int
    N2::Int
    N3::Int
    Nup1::Int
    Nup2::Int
    Nup3::Int
    Nnupts::Int

    # Fourier transform
    inplaceFFT::FFT
end

mutable struct Plan3{D,R<:FLT,C<:CPX,F<:CUFFT.cCuFFTPlan}
    plan1::Plan{D,R,C,F}
    plan2::Plan{D,R,C,F}
    nuFkernel::CuArray{R,1,cuDevBuffer}
    gamma1::R
    gamma2::R
    gamma3::R
end


#=
    makeplan for type 1,2
=#
function makeplan(nftype::Int, iflag::Int, ::Type{R},
    N::NTuple{D,Int}, Nnupts::Int; tol::Real=1e-6) #=
    =# where {D, R<:FLT}

    @assert (D in (1,2,3)) "Unsupported dimension: $D"
    @assert all(N .% 2 .== 0) "Only even number of pixels is supported."
    @assert (nftype in (1,2)) "Unknown nufft type: $nftype"
    @assert (iflag in (1,-1)) "Unknown iflag: $iflag"
    
    Nup = next235even.(2 .* N) # 2 .* N
    makeplan(nftype, iflag, R, N, Nup, Nnupts, tol=tol)
end

function makeplan(nftype::Int, iflag::Int, ::Type{R},
    N::NTuple{D,Int}, Nup::NTuple{D,Int}, Nnupts::Int; tol::Real=1e-6) where {D, R<:FLT}

    C = complex(R)
    
    N1 = N[1]; Nup1 = Nup[1]
    (D > 1) ? (N2 = N[2]; Nup2 = Nup[2]) : (N2 = 1; Nup2 = 1)
    (D > 2) ? (N3 = N[3]; Nup3 = Nup[3]) : (N3 = 1; Nup3 = 1)

    w::R = ceil(Int, -log10(tol/10.0))
    beta::R = 1.5π * w # 2.3

    upgrid = CuArray{C}(undef, Nup...)

    binSize1 = binSize2 = binSize3 = 1
    if D == 1
        binSize1 = 32
    elseif D == 2
        binSize1 = 32
        binSize2 = 32
    else
        binSize1 = 16
        binSize2 = 16
        binSize3 = 2
    end
    Nbin1 = ceil(Int, Nup1/binSize1)
    Nbin2 = (D > 1) ? ceil(Int, Nup2/binSize2) : 1
    Nbin3 = (D > 2) ? ceil(Int, Nup3/binSize3) : 1

    binCount    = CuArray{Int}(undef, Nbin1*Nbin2*Nbin3)
    binStartIdx = CuArray{Int}(undef, Nbin1*Nbin2*Nbin3)
    posIdxinBin = CuArray{Int}(undef, Nnupts)
    sortedIdx   = CuArray{Int}(undef, Nnupts)

    Fkernel1 = make_Fkernel(N1, Nup1, beta, w)
    Fkernel2 = (D > 1) ? make_Fkernel(N2, Nup2, beta, w) : CuArray{R}(undef,0)
    Fkernel3 = (D > 2) ? make_Fkernel(N3, Nup3, beta, w) : CuArray{R}(undef,0)

    inplaceFFT = (iflag > 0) ? CUDA.CUFFT.plan_bfft!(upgrid) : CUDA.CUFFT.plan_fft!(upgrid)
    FFTtype = typeof(inplaceFFT)

    Plan{D,R,C,FFTtype}(
        nftype, iflag, 
        w, beta, Fkernel1, Fkernel2, Fkernel3, 
        upgrid, 
        binSize1, binSize2, binSize3, Nbin1, Nbin2, Nbin3, 
        binCount, binStartIdx, posIdxinBin, sortedIdx, 
        N1, N2, N3, Nup1, Nup2, Nup3, Nnupts, 
        inplaceFFT
    )
end


#=
    makeplan for type 3
=#
function makeplan(nftype::Int, iflag::Int, ::Type{R},
    x::NTuple{D,CuArray{R,1}}, nu::NTuple{D,CuArray{R,1}}; tol::Real=1e-6) #=
    =# where {D, R<:FLT}

    @assert (D in (1,2,3)) "Unsupported dimension: $D"
    @assert (nftype == 3) "Unknown nufft type: $nftype"
    @assert (iflag in (1,-1)) "Unknown iflag: $iflag"

    Nnupts_x = length(x[1])
    Nnupts_nu = length(nu[1])
    for d = 2:D
        @assert Nnupts_x == length(x[d]) "Mismatch in the number of source pts"
        @assert Nnupts_nu == length(nu[d]) "Mismatch in the number of target pts"
    end
    
    X1 = X2 = X3 = one(R)
    Nu1 = Nu2 = Nu3 = one(R)
    X1 = maxabs(x[1])
    Nu1 = max(maxabs(nu[1]), 1/X1) # maybe 1/X1/(2π)
    if D > 1
        X2 = maxabs(x[2])
        Nu2 = max(maxabs(nu[2]), 1/X2)
    end
    if D > 2
        X3 = maxabs(x[3])
        Nu3 = max(maxabs(nu[3]), 1/X3)
    end

    w::R = ceil(Int, -log10(tol/10.0))
    beta::R = 1.5π * w # 2.3

    Nup1 = Nup2 = Nup3 = 1
    if D == 1
        Nup1 = next235even(ceil(Int, 8*X1*Nu1 + w))
        Nup = (Nup1,)
    elseif D == 2
        Nup1 = next235even(ceil(Int, 8*X1*Nu1 + w))
        Nup2 = next235even(ceil(Int, 8*X2*Nu2 + w))
        Nup = (Nup1, Nup2)
    else
        Nup1 = next235even(ceil(Int, 8*X1*Nu1 + w))
        Nup2 = next235even(ceil(Int, 8*X2*Nu2 + w))
        Nup3 = next235even(ceil(Int, 8*X3*Nu3 + w))
        Nup = (Nup1, Nup2, Nup3)
    end
    Nupup = 2 .* Nup # we need not call next235even here

    gamma1 = gamma2 = gamma3 = one(R)
    if D == 1
        gamma1 = Nup1 / (4*Nu1)
        gamma = (gamma1,)
    elseif D == 2
        gamma1 = Nup1 / (4*Nu1)
        gamma2 = Nup2 / (4*Nu2)
        gamma = (gamma1, gamma2)
    else
        gamma1 = Nup1 / (4*Nu1)
        gamma2 = Nup2 / (4*Nu2)
        gamma3 = Nup3 / (4*Nu3)
        gamma = (gamma1, gamma2, gamma3)
    end

    # Here, we use Nup for the argument "N" in makeplan for type 1.
    # It does not have any effect, since we use plan1 only for interpolation.
    # Dewindow is done in plan2 & execute! for type 3
    plan1 = makeplan(1, iflag, R, Nup, Nup, Nnupts_x, tol=tol)
    plan2 = makeplan(2, iflag, R, Nup, Nupup, Nnupts_nu, tol=tol)

    # This is a bit dirty, but...
    nu_cpu = Array{R}(undef, Nnupts_nu)
    nuFkernel_cpu = Array{R}(undef, Nnupts_nu)
    for d = 1:D
        copyto!(nu_cpu, nu[d])
        if d == 1
            nuFkernel_cpu .= make_Fkernel.(nu_cpu, Nup[d], beta, w, gamma[d])
        else
            nuFkernel_cpu .*= make_Fkernel.(nu_cpu, Nup[d], beta, w, gamma[d])
        end
    end
    nuFkernel = CuArray(nuFkernel_cpu)

    C = complex(R)
    FFTtype = typeof(plan1.inplaceFFT)
    Plan3{D,R,C,FFTtype}(plan1, plan2, nuFkernel, gamma1, gamma2, gamma3)
end