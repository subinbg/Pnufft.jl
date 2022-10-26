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
                        # Here, set the target tolerance loosely
                        rtol = (fp==Float32) ? 1e-3 : 10*tol

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