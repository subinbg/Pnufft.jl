# https://arxiv.org/abs/1912.09746, p. 26, eq. 5.21
function sinh_kernel(x::T, a::T, beta::T)::T where T<:FLT 
    scaled = x/a
    result = zero(T)
    if CUDA.abs(scaled) < 1
        result += CUDA.sinh( beta*CUDA.sqrt(one(T) - scaled*scaled) ) / CUDA.sinh(beta)
    end
    return result
end

function F_sinh_kernel(nu::Real, beta::F, w::F, Nup::Integer) where F <: FLT
    factor::F = beta*beta - (2*π*w*nu/Nup)^2
    
    _zero = zero(F)
    _one = one(Cint) # enforce openlibm
    result = zero(F)
    if factor > _zero
        result = besseli(_one, sqrt(factor)) / sqrt(factor) 
    elseif factor == _zero
        result = 0.5
    else
        result = besselj(_one, sqrt(-factor)) / sqrt(-factor)
    end
    return result
end

function make_Fkernel(N::Integer, Nup::Integer, beta::R, w::R) where R <: FLT
    Nhalf = N÷2
    Fkernel = Array{R}(undef, N)
    for i = 1:N
        @inbounds begin
            q = i-1-Nhalf
            minus1_q = (q % 2 == 0) ? 1 : -1
            Fkernel[i] = minus1_q * F_sinh_kernel(q, beta, w, Nup)
        end
    end
    # 1/(sigma*N = Nup) is omitted here,
    # because it will be omitted in the uniform FFT in type 1,2
    return CuArray(Fkernel) .* (w*π*beta/sinh(beta))
end

function make_Fkernel(nu::Real, Nup::Integer, beta::R, w::R, gamma::R) where R <: FLT
    nu_mod = nu * gamma
    # 1/(sigma*N = Nup) is omitted here (see, make_Fkernel for type 1,2)
    # and we omit (-1)^j for the shift in Fourier space
    # because we use NUFFT type 2 instead of uniform FFT in type 3.
    F_sinh_kernel(nu_mod, beta, w, Nup) * (w*π*beta/sinh(beta))
end