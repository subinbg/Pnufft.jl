macro gen_interp(D, direction)
    #=
        direction > 0: NU_to_U
        direction < 0: U_to_NU
    =#
    local func_name = (direction > 0) ? :interp_NU_to_U! : :interp_U_to_NU!

    local args = []
    local contents = []

    #= 
        Function arguments
    =# 
    push!( args, :($(:Nnupts) :: Int) )
    arg_expr_helper!(args, D, "Nup", :(Int))
    arg_expr_helper!(args, D, "x", :(CuDeviceArray{R,1}))
    push!( args, :($(:w) :: R) )
    push!( args, :($(:beta) :: R) )
    push!( args, :($(:sortedIdx) :: CuDeviceArray{Int,1}) )
    push!( args, :($(:nuC) :: CuDeviceArray{C,1}) )
    if direction > 0
        # uC: reinterpreted from CPX to FLT
        push!( args, :($(:uC) :: CuDeviceArray{R,$D}) )
    else
        push!( args, :($(:uC) :: CuDeviceArray{C,$D}) )
    end

    #= 
        Function contents
    =#
    local comm = :( idx = (blockIdx().x-1) * blockDim().x + threadIdx().x )
    push!(contents, comm)
    local comm = :( stride = blockDim().x * gridDim().x )
    push!(contents, comm)
    push!(contents, :(halfone::R = 0.5))
    
    local inside_loop = []
    push!( inside_loop, :(sortedIdx_ = sortedIdx[i]) )
    for d = 1:D # Rescale x from [-0.5,0.5] to [0, Nup]
        push!( inside_loop, :($(Symbol("x",d,"_rescaled")) = #=
        =# $(Symbol("Nup",d)) * ( $(Symbol("x",d))[sortedIdx_] + halfone)) ) #rescale(x1[sortedIdx[i]], Nup1)
    end

    for d = 1:D
        push!( inside_loop, :($(Symbol("x",d,"start")) = #=
        =# ceil(Int, $(Symbol("x",d,"_rescaled")) - w)) )

        push!( inside_loop, :($(Symbol("x",d,"end")) = #=
        =# floor(Int, $(Symbol("x",d,"_rescaled")) + w)) )
    end

    if direction < 0
        push!( inside_loop, :(source_re::R = zero(R)) )
        push!( inside_loop, :(source_im::R = zero(R)) )
    end
    
    inner_most = []
    push!( inner_most, :(distx1::R = abs(x1_rescaled - x1)) )
    push!( inner_most, :(ker1val = sinh_kernel(distx1, w, beta)) )
    push!( inner_most, Expr(:(=), :kerval, Expr(:call, :*, [Symbol("ker",d,"val") for d in 1:D]...)) )

    for d = 1:D # Periodic folding (periodic convolution)
        negative_shift = :($(Symbol("x",d)) + $(Symbol("Nup",d)))
        positive_shift = :(($(Symbol("x",d)) > $(Symbol("Nup",d))-1 ? $(Symbol("x",d))-$(Symbol("Nup",d)) : $(Symbol("x",d))))
        push!( inner_most, :($(Symbol("ix",d)) = #=
        =# ($(Symbol("x",d)) < 0) ? $negative_shift : $positive_shift ) )
    end

    # Julia is col-major; row is fastest, z is slowest
    # Index on upsampled uniform grid
    if D == 1
        push!(inner_most, :(uidx = ix1 + 1))
    elseif D == 2
        push!(inner_most, :(uidx = (ix1+1) + Nup1*ix2))
    else
        push!(inner_most, :(uidx = (ix1+1) + Nup1*ix2 + Nup1*Nup2*ix3))
    end

    if direction > 0 
        # Convolution
        # uC: reinterpreted from CPX to FLT
        push!(inner_most, :(source_re = nuC[sortedIdx_].re*kerval)) 
        push!(inner_most, :(source_im = nuC[sortedIdx_].im*kerval)) 
        push!(inner_most, :(CUDA.atomic_add!(pointer(uC, 2*uidx-1), source_re)))
        push!(inner_most, :(CUDA.atomic_add!(pointer(uC, 2*uidx), source_im)))
    else
        push!(inner_most, :(source_re += uC[uidx].re*kerval))
        push!(inner_most, :(source_im += uC[uidx].im*kerval))
    end

    loop = Expr(:for, :(x1 = x1start:x1end), quote $(inner_most...) end)
    if D > 1
        lv2_ker = []
        push!( lv2_ker, :(distx2::R = abs(x2_rescaled - x2)) )
        push!( lv2_ker, :(ker2val = sinh_kernel(distx2, w, beta)) )
        loop = Expr(:for, :(x2 = x2start:x2end), #=
        =# quote 
            $(lv2_ker...) 
            $(loop)
        end)
    end
    if D > 2
        lv3_ker = []
        push!( lv3_ker, :(distx3::R = abs(x3_rescaled - x3)) )
        push!( lv3_ker, :(ker3val = sinh_kernel(distx3, w, beta)) )
        loop = Expr(:for, :(x3 = x3start:x3end), #=
        =# quote 
            $(lv3_ker...) 
            $(loop)
        end)
    end

    inbounded = Expr(:macrocall, Symbol("@inbounds"), LineNumberNode(0), #=
    =# if direction > 0
            quote
                $(inside_loop...)
                $(loop)
            end
        else
            quote
                $(inside_loop...)
                $(loop)
                nuC[sortedIdx_] = source_re + im * source_im
            end
        end)

    push!(contents, Expr(:for, :(i = idx:stride:Nnupts), inbounded))
    push!(contents, :(return nothing))

    local func_def = quote
        function $(func_name)($(args...)) where {R<:FLT,C<:CPX}
            $(contents...)
        end
    end

    return esc(func_def)
end

# @macroexpand @gen_interp 3 1
@gen_interp 1 1
@gen_interp 2 1
@gen_interp 3 1
@gen_interp 1 -1
@gen_interp 2 -1
@gen_interp 3 -1


function interpolate!(
    plan::Plan{D,R,C}, 
    nuC::CuArray{C,1}, 
    x::Vararg{CuArray{R,1},D}; 
    threads::Int=256) #=
    =# where {D,R<:FLT,C<:CPX}

    nftype = plan.nftype
    Nnupts = plan.Nnupts
    Nup = [plan.Nup1, plan.Nup2, plan.Nup3][1:D]

    if nftype == 1
        uC_ = reinterpret(R, plan.upgrid)
        @cuda threads=threads blocks=(Nnupts+threads-1)÷threads #=
        =# interp_NU_to_U!(
            Nnupts,
            Nup...,
            x...,
            plan.w, plan.beta,
            plan.sortedIdx,
            nuC,
            uC_
        )
    else
        @cuda threads=threads blocks=(Nnupts+threads-1)÷threads #=
        =# interp_U_to_NU!(
            Nnupts,
            Nup...,
            x...,
            plan.w, plan.beta,
            plan.sortedIdx,
            nuC,
            plan.upgrid
        )
    end
end