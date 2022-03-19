const FLT = Union{Float32, Float64}
const CPX = Union{ComplexF32, ComplexF64}

using Pnufft
using CUDA
using BenchmarkTools
using Test


function prepare_data(::Type{T}, nftype::Int, Nnupts::Int, N::Vararg{Int,D}) where{T<:FLT,D}
    CPX = complex(T)
    two::T = 2.0

    X = Tuple(CuArray(rand(T, Nnupts) .- 1/two) for d in 1:D)
    S = Tuple(N[d]÷2 .* CuArray(two .* rand(T, *(N...)) .- one(T)) for d in 1:D)

    c = CuArray(two .* rand(CPX, Nnupts) .- one(T))
    ftilde_sz = nftype==3 ? (*(N...),) : N
    ftilde = CuArray(two .* rand(complex(T), ftilde_sz...) .- one(T))

    return X, S, c, ftilde
end

function validate(nftype::Int, iflag::Int, 
    X::NTuple{D,CuArray{R,1}}, S::NTuple{D,CuArray{R,1}}, 
    c::CuArray{C,1}, ftilde::CuArray{C,D2};
    nsamples::Int=20) where {R<:FLT,C<:CPX,D,D2}

    twopii::C = 2π * iflag * im
    X_cpu = Array.(X)
    S_cpu = Array.(S)
    c_cpu = Array(c)
    ftilde_cpu = Array(ftilde)

    grt = Array{C}(undef, nsamples)
    est = Array{C}(undef, nsamples)

    N = size(ftilde)
    Nnupts = length(c)
    for sample in 1:nsamples
        if nftype == 1
            nt = Tuple(floor(Int, rand()*N[d]) - N[d]÷2 for d in 1:D)
            manual = sum(c_cpu .* exp.(twopii.*(.+((nt .* X_cpu)...))))
            vnufft = ftilde_cpu[(N.÷2 .+nt .+1)...]

        elseif nftype == 2
            m::Int = floor(Int, rand()*Nnupts*0.9) + 1

            vnufft = c_cpu[m]
            ftilde_cpu_copy = deepcopy(ftilde_cpu)
            for d in 1:D
                shp = circshift(cat([N[d]], ones(Int, D-1), dims=1), d-1)
                ftilde_cpu_copy .*= exp.(twopii * (reshape(0:(N[d]-1), shp...).-N[d]÷2) * X_cpu[d][m])
            end
            manual = sum(ftilde_cpu_copy)
        else
            m2::Int = floor(Int, rand()*length(ftilde_cpu)*0.9) + 1

            vnufft = ftilde_cpu[m2]
            c_cpu_copy = deepcopy(c_cpu)
            for d in 1:D
                c_cpu_copy .*= exp.(twopii*S_cpu[d][m2]*X_cpu[d])
            end
            manual = sum(c_cpu_copy)
        end

        grt[sample] = manual
        est[sample] = vnufft
    end
    return grt, est
end


function run(nftype::Int, iflag::Int, 
    X::NTuple{D,CuArray{R,1}}, S::NTuple{D,CuArray{R,1}}, 
    c::CuArray{C,1}, ftilde::CuArray{C,D2};
    tol::Float64=1e-6, timecheck::Bool=false) where {R<:FLT,C<:CPX,D,D2}

    bench = nothing
    if nftype == 3
        if timecheck
            bench = @benchmark begin
                CUDA.@sync begin
                    plan = makeplan($nftype, $iflag, $R, $X, $S, tol=$tol)
                    execute!(plan, $c, $ftilde, $X..., $S..., threads=256, safecheck=false)
                end
            end
        else
            plan = makeplan(nftype, iflag, R, X, S, tol=tol)
            execute!(plan, c, ftilde, X..., S..., threads=256, safecheck=false)
        end
    else
        if timecheck
            bench = @benchmark begin
                CUDA.@sync begin
                    plan = makeplan($nftype, $iflag, $R, size($ftilde), length($c), tol=$tol)
                    execute!(plan, $c, $ftilde, $X..., threads=256, safecheck=false)
                end
            end
        else
            plan = makeplan(nftype, iflag, R, size(ftilde), length(c), tol=tol)
            execute!(plan, c, ftilde, X..., threads=256, safecheck=false)
        end
    end

    return bench
end


const N1,N2,N3 = 124,98,116 # 64,64,64 #
const tol = 1e-6
const dobench = false
nsamples = 3

@testset "Pnufft.jl" begin
    if CUDA.functional()
        for d in 1:3
            for M in (100, 1000, 10000, 100000) #, 1000000, 10000000
                for fp in (Float32, Float64)
                    @testset "($d,$M,$fp,$nftype)" for nftype in 1:3
                        N = (N1,N2,N3)[1:d]
                        x,s,c,ftilde = prepare_data(fp,nftype,M,N...)

                        # Low precision: rounding-off error occurs
                        # Type 3 transform: error accumulates
                        rtol = (fp==Float32) ? 5e-4 : (nftype == 3 ? sqrt(tol) : tol)
                        for iflag in (-1,1)
                            bench = run(nftype, iflag, x, s, c, ftilde, tol=tol, timecheck=false)

                            grt, est = validate(nftype, iflag, x, s, c, ftilde, nsamples=nsamples)
                            @test all(isapprox.(grt, est, rtol=rtol))
                        end
                    end
                end
            end
        end
    else
        @test true
    end
end


if dobench
    println()
    println("====== Benchmark ======")
    for d in 1:3
        println("Dimension $d:")
        N = (N1,N2,N3)[1:d]
        for M in (100, 1000, 10000, 100000, 1000000)
            for fp in (Float32, Float64)
                for nftype in 1:3
                    x,s,c,ftilde = prepare_data(fp,nftype,M,N...)
                    for iflag in (-1,1)
                        bench = run(nftype, iflag, x, s, c, ftilde, tol=tol, timecheck=true)

                        medtime = BenchmarkTools.prettytime(time(median(bench)))
                        direction = (iflag > 0) ? "+" : "-"
                        println("(M,fp,nftype,iflag) = ($M,$fp,$nftype,$direction): $medtime")
                    end
                end
            end
        end
        println()
    end
end