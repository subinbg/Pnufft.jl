"""
direction > 0: NU_to_U
direction < 0: U_to_NU
"""
macro gen_interp(D, direction, gpu)

    # uC: reinterpreted from CPX to FLT if NU_to_U
    local uC = direction > 0 ? :(uC ::AbstractArray{<:Real}) : :(uC ::AbstractArray{<:Complex})

    local x_rescale = [:(sortedIdx_ = sortedIdx[i])]
    for d = 1:D # Rescale x from [-0.5,0.5] to [0, Nup]
        x_d_rescale = var(:x, d, :_rescaled)
        Nup_d = var(:Nup, d)
        push!(x_rescale, :($x_d_rescale = rescale(x[sortedIdx_,$d], $Nup_d)))
    end

    local x_range = []
    for d = 1:D
        x_d_start = var(:x, d, :start)
        x_d_end = var(:x, d, :end)
        x_d_rescaled = var(:x, d, :_rescaled)

        push!(x_range, :($x_d_start = ceil(Int, $x_d_rescaled - m)))
        push!(x_range, :($x_d_end = floor(Int, $x_d_rescaled + m)))
    end

    local init_sources = []
    if direction < 0
        push!( init_sources, :(source_re::T = zero(T)) )
        push!( init_sources, :(source_im::T = zero(T)) )
    end
    
    

    # Periodic folding (periodic convolution)
    local pbc = []
    for d = 1:D 
        x_d_r = var(:x, d, :r)
        ix_d = var(:ix, d)
        Nup_d = var(:Nup, d)
    
        negative_shift = :($x_d_r + $Nup_d)
        positive_shift = :($x_d_r >= $Nup_d ? $x_d_r - $Nup_d : $x_d_r)
        push!(pbc, :($ix_d = $x_d_r < 0 ? $negative_shift : $positive_shift))
    end

    # Julia is col-major; row is the fastest, z is the slowest
    # Index on upsampled uniform grid
    local upgrid_scalar_index = [
        :(uidx = ix1 + 1),
        :(uidx = (ix1+1) + Nup1*ix2),
        :(uidx = (ix1+1) + Nup1*ix2 + Nup1*Nup2*ix3)
    ][D]

    local conv = []
    if direction > 0 
        # uC: reinterpreted from CPX to FLT
        push!(conv, :(source_re = nuC[sortedIdx_].re*kerval)) 
        push!(conv, :(source_im = nuC[sortedIdx_].im*kerval)) 
        push!(conv, :(atomic_add!(uC, 2*uidx-1, source_re))) # CUDA.atomic_add!(pointer(uC, 2*uidx-1), source_re)
        push!(conv, :(atomic_add!(uC, 2*uidx, source_im))) # CUDA.atomic_add!(pointer(uC, 2*uidx), source_im)
    else
        push!(conv, :(source_re += uC[uidx].re*kerval))
        push!(conv, :(source_im += uC[uidx].im*kerval))
    end

    kerval = Expr(:(=), :kerval, Expr(:call, :*, [var(:ker,d,:val) for d in 1:D]...))
    loop = [kerval, upgrid_scalar_index, conv...]
    # loop = quote
    #     $kerval
    #     $upgrid_scalar_index
    #     $(conv...)
    # end
    
    sinh_kernel = gpu ? :sinh_kernel_cuda : :sinh_kernel
    for d = 1:D
        distx_d = var(:distx,d)
        ker_d_val = var(:ker,d,:val)
        x_d_start = var(:x,d,:start)
        x_d_end = var(:x,d,:end)
        x_d_r = var(:x,d,:r)
        x_d_rescaled = var(:x,d,:_rescaled)

        dist = :($distx_d::T = abs($x_d_rescaled - $x_d_r))
        kerdval = :($ker_d_val = $(sinh_kernel)($distx_d, m, beta))

        comm = d == 1 ? [pbc[d], dist, kerdval, loop...] : [pbc[d], dist, kerdval, loop]
        loop = Expr(:for, :($x_d_r = $(x_d_start):$(x_d_end)), #=
        =# quote 
            $(comm...)
        end)
        # loop = quote
        #     for $x_d_r = $(x_d_start):$(x_d_end)
        #         $(pbc[d])
        #         $dist
        #         $kerdval
        #         $loop
        #     end
        # end
    end

    to_nucoeff = []
    if direction < 0
        push!(to_nucoeff, :(nuC[sortedIdx_] = source_re + im * source_im))
    end

    x = gpu ? :(x::CuDeviceArray{T}) : :(x::Array{T})
    local func_def = quote
        function interp!(Nnupts, $(argGen(:Nup, 1:D)...), sortedIdx,
            m, beta, nuC, $uC, $x) where T

            $(parallelize(
                quote
                    $(x_rescale...)
                    $(x_range...)
                    $(init_sources...)
                    $loop
                    $(to_nucoeff...)
                end,
                :Nnupts, gpu)...)

            return nothing
        end
    end

    return esc(func_def)
end

# @macroexpand @gen_interp 3 1 true
for d = (1,2,3)
    for direction = (1,-1)
        for gpu = (true, false)
            @eval @gen_interp $d $direction $gpu
        end
    end
end


function interpolate!(plan::NuFFTPlan{T,D,FK}, nuC, upgrid, nftype::Pair{Symbol,Symbol}) where {T,D,FK} 
    auto_config = FK <: CuArray ? auto_config_gpu : auto_config_cpu

    if nftype == (:nu => :u)
        uC = reinterpret(T, upgrid)
    else
        uC = upgrid
    end

    auto_config(
        plan.Nnupts,
        interp!,
        plan.Nnupts,
        plan.Nup...,
        plan.sortedIdx,
        plan.m,
        plan.beta,
        nuC,
        uC,
        plan.x
    )
end
