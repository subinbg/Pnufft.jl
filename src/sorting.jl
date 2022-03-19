# https://discourse.julialang.org/t/atomic-operations-issue-on-staticarrays-with-cudanative/39576
macro gen_BinCount(D)
    local args = []
    local contents = []

    #= 
        Function arguments
    =# 
    local varname_type = :( $(:Nnupts) :: Int  )
    push!(args, varname_type)

    arg_expr_helper!(args, D, "Nup", :(Int))
    arg_expr_helper!(args, D, "binSize", :(Int))
    arg_expr_helper!(args, D, "Nbin", :(Int))
    arg_expr_helper!(args, D, "x", :(CuDeviceArray{R,1}))

    push!( args, :($(:binCount) :: CuDeviceArray{Int,1}) )
    push!( args, :($(:posIdxinBin) :: CuDeviceArray{Int,1}) )


    #= 
        Function contents
    =#
    local comm = :( idx = (blockIdx().x-1) * blockDim().x + threadIdx().x )
    push!(contents, comm)
    local comm = :( stride = blockDim().x * gridDim().x )
    push!(contents, comm)
    push!(contents, :(halfone::R = 0.5))
    
    local inside_loop = []
    for d = 1:D # Rescale x from [-0.5,0.5] to [0, Nup]
        push!( inside_loop, :($(Symbol("x",d,"_rescaled")) = #=
        =# $(Symbol("Nup",d)) * ( $(Symbol("x",d))[i] + halfone)) ) #rescale(x1[i], Nup1)
    end

    for d = 1:D
        push!( inside_loop, :($(Symbol("bin",d)) = #=
        =# floor(Int, $(Symbol("x",d,"_rescaled"))/$(Symbol("binSize",d)))) )
    end

    for d = 1:D
        push!( inside_loop, :($(Symbol("bin",d)) = #=
        =# ($(Symbol("bin",d)) >= $(Symbol("Nbin",d))) ? $(Symbol("bin",d))-1 : $(Symbol("bin",d))) )
    end

    # Julia is col-major; row is fastest, z is slowest
    if D == 1
        push!(inside_loop, :(binidx = bin1 + 1))
    elseif D == 2
        push!(inside_loop, :(binidx = (bin1+1) + Nbin1*bin2))
    else
        push!(inside_loop, :(binidx = (bin1+1) + Nbin1*bin2 + Nbin1*Nbin2*bin3))
    end

    push!(inside_loop, :(idxInsideBin = CUDA.atomic_add!(pointer(binCount, binidx), 1))) # binCount[binidx]
    push!(inside_loop, :(posIdxinBin[i] = idxInsideBin + 1)) # since idx starts at 1 in Julia

    local inside_loop_quote = quote
        $(inside_loop...)
    end
    inbounded = Expr(:macrocall, Symbol("@inbounds"), LineNumberNode(0), inside_loop_quote)

    push!(contents, Expr(:for, :(i = idx:stride:Nnupts), inbounded))
    push!(contents, :(return nothing))

    local func_def = quote
        function BinCount!($(args...)) where R<:FLT
            $(contents...)
        end
    end

    return esc(func_def)
end


macro gen_AssignPosIdx(D)
    local args = []
    local contents = []

    #= 
        Function arguments
    =# 
    local varname_type = :( $(:Nnupts) :: Int  )
    push!(args, varname_type)

    arg_expr_helper!(args, D, "Nup", :(Int))
    arg_expr_helper!(args, D, "binSize", :(Int))
    arg_expr_helper!(args, D, "Nbin", :(Int))
    arg_expr_helper!(args, D, "x", :(CuDeviceArray{R,1}))

    push!( args, :($(:binStartIdx) :: CuDeviceArray{Int,1}) )
    push!( args, :($(:posIdxinBin) :: CuDeviceArray{Int,1}) )
    push!( args, :($(:sortedIdx) :: CuDeviceArray{Int,1}) )


    #= 
        Function contents
    =#
    local comm = :( idx = (blockIdx().x-1) * blockDim().x + threadIdx().x )
    push!(contents, comm)
    local comm = :( stride = blockDim().x * gridDim().x )
    push!(contents, comm)
    push!(contents, :(halfone::R = 0.5))

    local inside_loop = []
    for d = 1:D # Rescale x from [-0.5,0.5] to [0, Nup]
        push!( inside_loop, :($(Symbol("x",d,"_rescaled")) = #=
        =# $(Symbol("Nup",d)) * ( $(Symbol("x",d))[i] + halfone)) ) #rescale(x1[i], Nup1)
    end

    for d = 1:D
        push!( inside_loop, :($(Symbol("bin",d)) = #=
        =# floor(Int, $(Symbol("x",d,"_rescaled"))/$(Symbol("binSize",d)))) )
    end

    for d = 1:D
        push!( inside_loop, :($(Symbol("bin",d)) = #=
        =# ($(Symbol("bin",d)) >= $(Symbol("Nbin",d))) ? $(Symbol("bin",d))-1 : $(Symbol("bin",d))) )
    end

    # Julia is col-major; row is fastest, z is slowest
    if D == 1
        push!(inside_loop, :(binidx = bin1 + 1))
    elseif D == 2
        push!(inside_loop, :(binidx = (bin1+1) + Nbin1*bin2))
    else
        push!(inside_loop, :(binidx = (bin1+1) + Nbin1*bin2 + Nbin1*Nbin2*bin3))
    end

    push!(inside_loop, :(sortedIdx[binStartIdx[binidx]+posIdxinBin[i]] = i))

    local inside_loop_quote = quote
        $(inside_loop...)
    end
    inbounded = Expr(:macrocall, Symbol("@inbounds"), LineNumberNode(0), inside_loop_quote)

    push!(contents, Expr(:for, :(i = idx:stride:Nnupts), inbounded))
    push!(contents, :(return nothing))

    local func_def = quote
        function AssignPosIdx!($(args...)) where R<:FLT
            $(contents...)
        end
    end

    return esc(func_def)
end

function SetStartIdx!(binStartIdx::CuDeviceArray{Int,1}, binCount::CuDeviceArray{Int,1}, Nbins::Int)
    idx = (blockIdx().x-1) * blockDim().x + threadIdx().x
    stride = blockDim().x * gridDim().x

    for i = idx:stride:Nbins
        @inbounds begin
            startIdx = 0
            for b = 2:i
                startIdx += binCount[b-1]
            end
            binStartIdx[i] = startIdx
        end
    end
end

# @macroexpand @gen_BinCount 1
@gen_BinCount 1
@gen_BinCount 2
@gen_BinCount 3

@gen_AssignPosIdx 1
@gen_AssignPosIdx 2
@gen_AssignPosIdx 3

function BinSort!(
    plan::Plan{D,R}, 
    x::Vararg{CuArray{R,1},D}; 
    threads::Int=256) #=
    =# where {D,R<:FLT}

    # Initializing counts
    fill!(plan.binCount, zero(Int))

    Nnupts = plan.Nnupts
    Nup = [plan.Nup1, plan.Nup2, plan.Nup3][1:D]
    binSize = [plan.binSize1, plan.binSize2, plan.binSize3][1:D]
    Nbin = [plan.Nbin1, plan.Nbin2, plan.Nbin3][1:D]

    @cuda threads=threads blocks=(Nnupts+threads-1)÷threads #=
    =# BinCount!(
        Nnupts,     # Number of non-uniform points
        Nup...,     # number of pixels in upsampled grid
        binSize..., # Number of pixels per bin
        Nbin...,    # Number of bins per dimension
        x...,       # Non-uniform coordinates
        plan.binCount,   # number of points in a bin = starting index of each bin
        plan.posIdxinBin # index of each position in a bin    
    )

    Nbins = *(Nbin...)
    @cuda threads=threads blocks=(Nbins+threads-1)÷threads #=
    =# SetStartIdx!(plan.binStartIdx, plan.binCount, Nbins)

    #=
        index of x:
          bin1    bin3
        ----------------
        |  1   |    4  |
        |    7 |   2   |
        ----------------
        |   3  |  8    |
        |  6   |   5   |
        ----------------
          bin2    bin4

        Then sortedIdx is, approximately, [1,7,3,6,4,2,8,5].
        Using sortedIdx is useful in interpolation.
        For example, in type 1, we interpolate non-uniform points to unifrom grid.
        After interpolating to uniform grid points adjacent to x[1],
        doing interpolation from x[7] will result in accessing uniform grid points
        similar to those from x[1],
        which significantly reduces memory access time.
    =#
    @cuda threads=threads blocks=(Nnupts+threads-1)÷threads #=
    =# AssignPosIdx!(
        Nnupts, 
        Nup...,
        binSize...,
        Nbin...,
        x...,
        plan.binStartIdx, # starting index of each bin
        plan.posIdxinBin, # index of each position in a bin   
        plan.sortedIdx    # sorted position index  
    )
end









