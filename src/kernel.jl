"""
https://arxiv.org/abs/1912.09746, p. 26, eq. 5.21
"""
function sinh_kernel_cuda(x::T, a, beta::T)::T where T
    scaled = x/a
    result = zero(T)
    if CUDA.abs(scaled) < 1
        result += CUDA.sinh( beta*CUDA.sqrt(one(T) - scaled*scaled) ) / CUDA.sinh(beta)
    end
    return result
end

function sinh_kernel(x::T, a, beta::T)::T where T
    scaled = x/a
    result = zero(T)
    if abs(scaled) < 1
        result += sinh( beta*sqrt(one(T) - scaled*scaled) ) / sinh(beta)
    end
    return result
end

function F_sinh_kernel(Nup, beta, m, nu)
    factor = beta*beta - (2*π*m*nu/Nup)^2
    
    _zero = zero(factor)
    _one = one(nu)
    result = zero(factor)
    if factor > _zero
        result = besseli(_one, sqrt(factor)) / sqrt(factor) 
    elseif factor == _zero
        result = 0.5
    else
        result = besselj(_one, sqrt(-factor)) / sqrt(-factor)
    end
    return result
end

function make_Fkernel(N::Integer, Nup::Integer, beta::T, m::Integer) where T
    Nhalf = N÷2
    Fkernel = Array{T}(undef, N)
    @inbounds for i = 1:N
        q = i-1-Nhalf
        minus1_q = (q % 2 == 0) ? 1 : -1
        Fkernel[i] = minus1_q * F_sinh_kernel(Nup, beta, m, q)
    end
    # 1/(sigma*N = Nup) is omitted here,
    # because it will be omitted in the uniform FFT in type 1,2
    Fkernel .*= (m*π*beta/sinh(beta))
    return Fkernel
end

function make_Fkernel(nu::Real, Nup::Integer, beta::T, m::Integer, gamma::T) where T
    nu_mod = nu * gamma
    # 1/(sigma*N = Nup) is omitted here (see, make_Fkernel for type 1,2)
    # and we omit (-1)^j for the shift in Fourier space
    # because we use NUFFT type 2 instead of uniform FFT in type 3.
    out::T = F_sinh_kernel(Nup, beta, m, nu_mod) * (m*π*beta/sinh(beta))
    return out
end