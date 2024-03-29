"""
Compute scalar index in D-dimensional bins as follows:    
- In each dimension, convert `x` from [-0.5, 0.5] to [0, Nup]
- Get bin index with range [0, Nbin) by dividing `x` with binSize defined in `plan`
- Finally, get the scalar bin index whose variable name is `binidx`
"""
function get_bin_index(D)
    x_rescale = []
    for d = 1:D # Rescale x from [-0.5,0.5] to [0, Nup]
        x_d_rescale = var(:x, d, :_rescaled)
        push!(x_rescale, :($x_d_rescale = rescale(x[i,$d], $(var(:Nup, d)))))
    end

    bin_index = []
    for d = 1:D
        bin_d = var(:bin, d)
        x_d_rescale = var(:x, d, :_rescaled)
        binSize_d = var(:binSize, d)

        push!(bin_index, :($bin_d = floor(Int, $x_d_rescale/$binSize_d)))
    end

    bin_index_adjust = []
    for d = 1:D
        bin_d = var(:bin, d)
        Nbin_d = var(:Nbin, d)
        push!(bin_index_adjust, :($bin_d -= $bin_d >= $Nbin_d ? 1 : 0))
    end

    # Julia is col-major; row is the fastest, z is the slowest
    bin_scalar_index = [
        :(binidx = bin1 + 1),
        :(binidx = (bin1+1) + Nbin1*bin2),
        :(binidx = (bin1+1) + Nbin1*bin2 + Nbin1*Nbin2*bin3)
    ][D]

    return x_rescale, bin_index, bin_index_adjust, bin_scalar_index
end


"""
- Store the number of `x` in each bin (indexed by `binidx`) in `binCount`
- Store the index of `x` in each bin in `posIdxinBin`
"""
macro gen_BinCount(D, gpu)

    x_rescale, bin_index, bin_index_adjust, bin_scalar_index = get_bin_index(D)

    x = gpu ? :x : :(x::Array)
    local func_def = quote
        function BinCount!(Nnupts, $(argGen(:Nup, 1:D)...), 
            $(argGen(:binSize, 1:D)...), $(argGen(:Nbin, 1:D)...),
            binCount, posIdxinBin, $x)
            
            $(parallelize(
                quote
                    $(x_rescale...)
                    $(bin_index...)
                    $(bin_index_adjust...)
                    $bin_scalar_index
                    
                    idxInsideBin = atomic_add!(binCount, binidx, 1)
                    posIdxinBin[i] = idxInsideBin + 1 # since idx starts at 1 in Julia
                end,
                :Nnupts, gpu)...)

            return nothing
        end
    end

    return esc(func_def)
end


"""
- Sum # of `x`-points in bins with indexes 1,2,...,i-1 
- Store the sum in `binStartIdx[i]`
"""
macro gen_SetStartIdx!(gpu)
    binCount = gpu ? :binCount : :(binCount::Array)

    local func_def = quote
        function SetStartIdx!(Nbins, binStartIdx, $binCount)
            $(parallelize(
                quote
                    startIdx = 0
                    for b = 2:i
                        startIdx += binCount[b-1]
                    end
                    binStartIdx[i] = startIdx
                end,
                :Nbins, gpu)...)
            return nothing
        end
    end
    return esc(func_def)
end
# function SetStartIdx!(binStartIdx, binCount) # This is incorrect
#     cumsum!(binStartIdx, binCount) 
#     binStartIdx .-= view(binStartIdx, 1)
# end


"""
`sortedIdx` contains point indexes such that 
points close to each other are also closely located in `sortedIdx`.
"""
macro gen_AssignPosIdx(D, gpu)
    x_rescale, bin_index, bin_index_adjust, bin_scalar_index = get_bin_index(D)

    x = gpu ? :x : :(x::Array)
    local func_def = quote
        function AssignPosIdx!(Nnupts, $(argGen(:Nup, 1:D)...), 
            $(argGen(:binSize, 1:D)...), $(argGen(:Nbin, 1:D)...),
            binStartIdx, posIdxinBin, sortedIdx, $x)

            $(parallelize(
                quote
                    $(x_rescale...)
                    $(bin_index...)
                    $(bin_index_adjust...)
                    $bin_scalar_index
                    
                    sortedIdx[binStartIdx[binidx]+posIdxinBin[i]] = i
                end,
                :Nnupts, gpu)...)

            return nothing
        end
    end

    return esc(func_def)
end

# @macroexpand @gen_BinCount 1 true
for gpu = (true, false)
    @eval (@gen_SetStartIdx! $gpu)
    for d = (1, 2, 3)
        @eval (@gen_BinCount $d $gpu)
        @eval (@gen_AssignPosIdx $d $gpu)
    end
end


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
=#
"""
In the above figure, sortedIdx is, approximately, [1,7,3,6,4,2,8,5].     
Using sortedIdx is useful in interpolation.     
For example, in type 1, we interpolate non-uniform points to unifrom grid.     
After interpolating to uniform grid points adjacent to x[1],     
doing interpolation from x[7] will result in accessing uniform grid points     
similar to those from x[1],     
which significantly reduces memory access time.     
"""
function BinSort!(plan::NuFFTPlan{T,D,FK}) where {T,D,FK} 
    auto_config = FK <: CuArray ? auto_config_gpu : auto_config_cpu

    # Initializing counts
    fill!(plan.binCount, zero(eltype(plan.binCount)))

    auto_config(
        plan.Nnupts, 
        BinCount!, 
        plan.Nnupts,      # Number of non-uniform points
        plan.Nup...,      # number of pixels in upsampled grid
        plan.binSize...,  # Number of pixels per bin
        plan.Nbin...,     # Number of bins per dimension
        plan.binCount,    # number of points in a bin = starting index of each bin
        plan.posIdxinBin, # index of each position in a bin    
        plan.x            # Non-uniform coordinates
    )

    Nbins = *(plan.Nbin...)
    SetStartIdx!(Nbins, plan.binStartIdx, plan.binCount)

    auto_config(
        plan.Nnupts, 
        AssignPosIdx!,
        plan.Nnupts,
        plan.Nup...,
        plan.binSize...,
        plan.Nbin...,
        plan.binStartIdx, # starting index of each bin
        plan.posIdxinBin, # index of each position in a bin   
        plan.sortedIdx,   # sorted position index  
        plan.x
    )
end