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









# """
# direction > 0: NU_to_U
# direction < 0: U_to_NU
# """
# macro gen_interp(D, direction)

#     local func_name = (direction > 0) ? :interp_NU_to_U! : :interp_U_to_NU!

#     local args = []
#     local contents = []

#     #= 
#         Function arguments
#     =# 
#     push!( args, :($(:Nnupts) :: Int) )
#     arg_expr_helper!(args, D, "Nup", :(Int))
#     arg_expr_helper!(args, D, "x", :(CuDeviceArray{R,1}))
#     push!( args, :($(:w) :: R) )
#     push!( args, :($(:beta) :: R) )
#     push!( args, :($(:sortedIdx) :: CuDeviceArray{Int,1}) )
#     push!( args, :($(:nuC) :: CuDeviceArray{C,1}) )
#     if direction > 0
#         # uC: reinterpreted from CPX to FLT
#         push!( args, :($(:uC) :: CuDeviceArray{R,$D}) )
#     else
#         push!( args, :($(:uC) :: CuDeviceArray{C,$D}) )
#     end

#     #= 
#         Function contents
#     =#
#     local comm = :( idx = (blockIdx().x-1) * blockDim().x + threadIdx().x )
#     push!(contents, comm)
#     local comm = :( stride = blockDim().x * gridDim().x )
#     push!(contents, comm)
#     push!(contents, :(halfone::R = 0.5))
    
#     local inside_loop = []
#     push!( inside_loop, :(sortedIdx_ = sortedIdx[i]) )
#     for d = 1:D # Rescale x from [-0.5,0.5] to [0, Nup]
#         push!( inside_loop, :($(Symbol("x",d,"_rescaled")) = #=
#         =# $(Symbol("Nup",d)) * ( $(Symbol("x",d))[sortedIdx_] + halfone)) ) #rescale(x1[sortedIdx[i]], Nup1)
#     end

#     for d = 1:D
#         push!( inside_loop, :($(Symbol("x",d,"start")) = #=
#         =# ceil(Int, $(Symbol("x",d,"_rescaled")) - w)) )

#         push!( inside_loop, :($(Symbol("x",d,"end")) = #=
#         =# floor(Int, $(Symbol("x",d,"_rescaled")) + w)) )
#     end

#     if direction < 0
#         push!( inside_loop, :(source_re::R = zero(R)) )
#         push!( inside_loop, :(source_im::R = zero(R)) )
#     end
    
#     inner_most = []
#     push!( inner_most, :(distx1::R = abs(x1_rescaled - x1)) )
#     push!( inner_most, :(ker1val = sinh_kernel(distx1, w, beta)) )
#     push!( inner_most, Expr(:(=), :kerval, Expr(:call, :*, [Symbol("ker",d,"val") for d in 1:D]...)) )

#     for d = 1:D # Periodic folding (periodic convolution)
#         negative_shift = :($(Symbol("x",d)) + $(Symbol("Nup",d)))
#         positive_shift = :(($(Symbol("x",d)) > $(Symbol("Nup",d))-1 ? $(Symbol("x",d))-$(Symbol("Nup",d)) : $(Symbol("x",d))))
#         push!( inner_most, :($(Symbol("ix",d)) = #=
#         =# ($(Symbol("x",d)) < 0) ? $negative_shift : $positive_shift ) )
#     end

#     # Julia is col-major; row is fastest, z is slowest
#     # Index on upsampled uniform grid
#     if D == 1
#         push!(inner_most, :(uidx = ix1 + 1))
#     elseif D == 2
#         push!(inner_most, :(uidx = (ix1+1) + Nup1*ix2))
#     else
#         push!(inner_most, :(uidx = (ix1+1) + Nup1*ix2 + Nup1*Nup2*ix3))
#     end

#     if direction > 0 
#         # Convolution
#         # uC: reinterpreted from CPX to FLT
#         push!(inner_most, :(source_re = nuC[sortedIdx_].re*kerval)) 
#         push!(inner_most, :(source_im = nuC[sortedIdx_].im*kerval)) 
#         push!(inner_most, :(CUDA.atomic_add!(pointer(uC, 2*uidx-1), source_re)))
#         push!(inner_most, :(CUDA.atomic_add!(pointer(uC, 2*uidx), source_im)))
#     else
#         push!(inner_most, :(source_re += uC[uidx].re*kerval))
#         push!(inner_most, :(source_im += uC[uidx].im*kerval))
#     end

#     loop = Expr(:for, :(x1 = x1start:x1end), quote $(inner_most...) end)
#     if D > 1
#         lv2_ker = []
#         push!( lv2_ker, :(distx2::R = abs(x2_rescaled - x2)) )
#         push!( lv2_ker, :(ker2val = sinh_kernel(distx2, w, beta)) )
#         loop = Expr(:for, :(x2 = x2start:x2end), #=
#         =# quote 
#             $(lv2_ker...) 
#             $(loop)
#         end)
#     end
#     if D > 2
#         lv3_ker = []
#         push!( lv3_ker, :(distx3::R = abs(x3_rescaled - x3)) )
#         push!( lv3_ker, :(ker3val = sinh_kernel(distx3, w, beta)) )
#         loop = Expr(:for, :(x3 = x3start:x3end), #=
#         =# quote 
#             $(lv3_ker...) 
#             $(loop)
#         end)
#     end

#     inbounded = Expr(:macrocall, Symbol("@inbounds"), LineNumberNode(0), #=
#     =# if direction > 0
#             quote
#                 $(inside_loop...)
#                 $(loop)
#             end
#         else
#             quote
#                 $(inside_loop...)
#                 $(loop)
#                 nuC[sortedIdx_] = source_re + im * source_im
#             end
#         end)

#     push!(contents, Expr(:for, :(i = idx:stride:Nnupts), inbounded))
#     push!(contents, :(return nothing))

#     local func_def = quote
#         function $(func_name)($(args...)) where {R<:FLT,C<:CPX}
#             $(contents...)
#         end
#     end

#     return esc(func_def)
# end