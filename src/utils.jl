const defaultBinSize = ((1024,), (32,32), (16,16,2))

function check_args(D, X, N, of, Nbin)
    @assert (D in (1,2,3)) "Unsupported dimension: $D"
    @assert size(X, 2) == D "N=$N, but non-uniform points are not $(D)-dimensional"
    @assert all(N .% 2 .== 0) "Only even number of pixels is supported."
    @assert 1.25 <= of <= 2.0 "Unsupported oversampling factor: $(of)"
    @assert all(Nbin .> 0) "Wrong binSize=$binSize for N=$N"
end

function rescale(x::T, N::Integer) where T 
    out::T = x*N + N÷2
end

function auto_config_gpu(N_target::Integer, f::Function, args...)
    kernel = @cuda launch=false f(args...)
    config = launch_configuration(kernel.fun)
    threads = min(N_target, config.threads)
    blocks = cld(N_target, threads)

    CUDA.@sync begin
        kernel(args...; threads, blocks)
    end
end
auto_config_cpu(N_target::Integer, f::Function, args...) = begin f(args...) end


var(args...) = Symbol(args...)
argGen(s::Symbol, r::UnitRange) = [var(s,_r) for _r in r]

function to_device(A::T, gpu::Bool) where T<:AbstractArray
    devfunc = gpu ? CuArray : Array
    out = T <: devfunc ? begin A_copy = similar(A); copyto!(A_copy, A) end : devfunc(A)
end

function parallelize(expr, Nmax, gpu::Bool)

    inbounded = Expr(:macrocall, Symbol("@inbounds"), LineNumberNode(0), expr)
    gpuidx = []
    if gpu
        push!(gpuidx, :(idx = (blockIdx().x-1) * blockDim().x + threadIdx().x))
        push!(gpuidx, :(stride = blockDim().x * gridDim().x))
    end

    parallelized = gpu ? 
        Expr(:for, :(i = idx:stride:$Nmax), inbounded) :
        # Expr(:for, :(i = 1:$Nmax), inbounded) # in case of serial atomic_add!
        Expr(:macrocall, Symbol("@floop"), LineNumberNode(0), :(ThreadedEx()), Expr(:for, :(i = 1:$Nmax), inbounded))

    return [gpuidx..., parallelized]
end
