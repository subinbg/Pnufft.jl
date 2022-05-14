using Pnufft
using CUDA
using Test
Pnufft.FFTW.set_num_threads(Threads.nthreads())

include("utils.jl")
const N1,N2,N3 = 64,64,64 # 124,98,116 # 
const nsamples = 5
const tol = 1e-6

@testset "Pnufft.jl" begin
    for d in 1:3
        for Nnupts = 10 .^ (2:4)
            for fp = (Float32, Float64)
                @testset "($d,$Nnupts,$fp,$nftype)" for nftype in 1:3
                    for cuda = (false, true)
                        if cuda && !CUDA.functional()
                            continue
                        end

                        X, S, c, ftilde = prepare_data(fp, nftype, Nnupts, (N1,N2,N3)[1:d]; dev=cuda ? gpu : cpu)
                        plan = nftype == 3 ? 
                            plan_nufft!(X, S, gpu=cuda, reltol=tol) : 
                            plan_nufft!(X, (N1,N2,N3)[1:d], gpu=cuda, reltol=tol)
                        source = nftype == 2 ? ftilde : c
                        target = nftype == 2 ? c : ftilde

                        # Low precision: rounding-off error occurs
                        rtol = (fp==Float32) ? 1e-3 : 3.0*tol

                        for iflag in (-1,1)
                            plan(target, iflag, source)
                            grt, est = validate(nftype, iflag, X, S, c, ftilde, nsamples=nsamples)
                            @test isapprox(grt, est, rtol=rtol)
                        end
                    end
                end
            end
        end
    end
end


# function run(nftype::Int, iflag::Int, 
#     X::AbstractArray{T,2}, S::AbstractArray{T,2}, 
#     c::AbstractArray{Complex{T},1}, ftilde::AbstractArray{Complex{T},D};
#     tol::Float64=1e-6, timecheck::Bool=false) where {T,D}

#     bench = nothing
#     if nftype == 3
#         if timecheck
#             bench = @benchmark begin
#                 CUDA.@sync begin
#                     plan = makeplan($nftype, $iflag, $R, $X, $S, tol=$tol)
#                     execute!(plan, $c, $ftilde, $X..., $S..., threads=256, safecheck=false)
#                 end
#             end
#         else
#             plan = makeplan(nftype, iflag, R, X, S, tol=tol)
#             execute!(plan, c, ftilde, X..., S..., threads=256, safecheck=false)
#         end
#     else
#         if timecheck
#             bench = @benchmark begin
#                 CUDA.@sync begin
#                     plan = plan_nufft!($X, size($ftilde), reltol=$tol)
#                     set_iflag!(plan, $iflag)
#                     init!(plan)
#                     source = $nftype == 1 ? $c : $ftilde
#                     target = $nftype == 1 ? $ftilde : $c
#                     plan(target, source)
#                 end
#             end
#         else
#             plan = plan_nufft!(X, size(ftilde), reltol=tol)
#             set_iflag!(plan, iflag)
#             init!(plan)
#             source = nftype == 1 ? c : ftilde
#             target = nftype == 1 ? ftilde : c
#             plan(target, source)
#         end
#     end

#     return bench
# end





# if dobench
#     println()
#     println("====== Benchmark ======")
#     for d in 1:3
#         println("Dimension $d:")
#         N = (N1,N2,N3)[1:d]
#         for M in (100, 1000, 10000, 100000, 1000000)
#             for fp in (Float32, Float64)
#                 for nftype in 1:3
#                     x,s,c,ftilde = prepare_data(fp,nftype,M,N...)
#                     for iflag in (-1,1)
#                         bench = run(nftype, iflag, x, s, c, ftilde, tol=tol, timecheck=true)

#                         medtime = BenchmarkTools.prettytime(time(median(bench)))
#                         direction = (iflag > 0) ? "+" : "-"
#                         println("(M,fp,nftype,iflag) = ($M,$fp,$nftype,$direction): $medtime")
#                     end
#                 end
#             end
#         end
#         println()
#     end
# end