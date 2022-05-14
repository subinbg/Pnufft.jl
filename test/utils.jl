cpu(A::T) where T = T <: Array ? A : Array(A)
gpu(A::T) where T = T <: CuArray ? A : CuArray(A)

function prepare_data(::Type{T}, nftype::Int, Nnupts::Integer, N::NTuple{D,Int}; dev=cpu) where {T,D}
    X = rand(T, Nnupts, D) .- one(T)/2 |> dev
    S = reshape([N[d]÷2 for d in 1:D], 1, D) .* (2 .* rand(T, *(N...), D) .- one(T)) |> dev

    c = 2 .* rand(Complex{T}, Nnupts) .- one(T) |> dev
    ftilde_sz = nftype == 3 ? (*(N...),) : N
    ftilde = 2 .* rand(complex(T), ftilde_sz...) .- one(T) |> dev

    return X, S, c, ftilde
end

function val_type1(twopii::Complex, c::Array, ftilde::Array{T,D}, X::Array, S::Array) where {T,D}
    N = size(ftilde)
    nt = reshape([floor(Int, rand()*N[d]) - N[d]÷2 for d in 1:D], 1, D)

    grt = sum(c .* exp.(twopii.*(sum(nt .* X, dims=2))))
    est = ftilde[(N.÷2 .+ Tuple(nt[1:D]) .+ 1)...]
    return grt, est
end

function val_type2(twopii::Complex, c::Array, ftilde::Array{T,D}, X::Array, S::Array) where {T,D}
    N = size(ftilde)
    Nnupts = length(c)
    m = floor(Int, rand()*Nnupts*0.9) + 1

    est = c[m]
    ftilde_cp = copy(ftilde)
    for d in 1:D
        shp = circshift(cat([N[d]], ones(Int, D-1), dims=1), d-1)
        nx = reshape((Array(0:(N[d]-1)).-N[d]÷2) .* X[m,d], shp...)
        ftilde_cp .*= exp.(twopii .* nx)
    end
    grt = sum(ftilde_cp)

    return grt, est
end

function val_type3(twopii::Complex, c::Array, ftilde::Array, X::Array, S::Array)
    m = floor(Int, rand()*length(ftilde)*0.9) + 1
    D = size(X,2)

    est = ftilde[m]
    c_cp = copy(c)
    for d in 1:D
        c_cp .*= exp.(twopii.*S[m,d].*X[:,d])
    end
    grt = sum(c_cp)

    return grt, est
end

function validate(nftype::Int, iflag::Int, 
    X::AbstractArray{T,2}, S::AbstractArray{T,2}, 
    c::AbstractArray{Complex{T},1}, ftilde::AbstractArray{Complex{T}};
    nsamples::Int=20) where T

    twopii::Complex{T} = 2π * iflag * im
    
    X_cpu = X |> cpu
    S_cpu = S |> cpu
    c_cpu = c |> cpu
    ftilde_cpu = ftilde |> cpu

    grt = Array{Complex{T}}(undef, nsamples)
    est = Array{Complex{T}}(undef, nsamples)

    val_func = eval(Symbol(:val_type, nftype))
    for sample in 1:nsamples
        gt_val, est_val = val_func(twopii, c_cpu, ftilde_cpu, X_cpu, S_cpu)

        grt[sample] = gt_val
        est[sample] = est_val
    end
    return grt, est
end