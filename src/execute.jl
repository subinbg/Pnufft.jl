function (plan::NuFFTPlan{T,D})(target::AbstractArray{Complex{T},Dt}, 
    iflag::Integer, source::AbstractArray{Complex{T},Ds}) where {T,D,Dt,Ds}

    if D == Dt && length(source) == plan.Nnupts && all(size(target) .== plan.N)
        execute_type1!(plan, target, iflag, source)
    elseif D == Ds && length(target) == plan.Nnupts && all(size(source) .== plan.N)
        execute_type2!(plan, target, iflag, source)
    else
        error("Unknown source/target combination")
    end
end


"""
Type 1: NU => U
"""
function execute_type1!(plan::NuFFTPlan{T,D}, target::AbstractArray{Complex{T},D}, 
    iflag::Integer, source::AbstractArray{Complex{T},1}) where {T,D}

    fill!(plan.upgrid[], zero(Complex{T}))
    interpolate!(plan, source, plan.upgrid[], :nu => :u)

    FT = iflag > 0 ? plan.IFFT : plan.FFT
    FT*plan.upgrid[]
    dewindow!(plan, plan.upgrid[], target, :nu => :u)
end

"""
Type 2: U => NU
"""
function execute_type2!(plan::NuFFTPlan{T,D}, target::AbstractArray{Complex{T},1}, 
    iflag::Integer, source::AbstractArray{Complex{T},D}) where {T,D}
    
    fill!(plan.upgrid[], zero(Complex{T}))
    dewindow!(plan, plan.upgrid[], source, :u => :nu)

    FT = iflag > 0 ? plan.IFFT : plan.FFT
    FT*plan.upgrid[]
    interpolate!(plan, target, plan.upgrid[], :u => :nu)
end


"""
Type 3: NU => NU
"""
function (plan::NuFFTPlan3{T})(target::AbstractArray{Complex{T},1}, 
    iflag::Integer, source::AbstractArray{Complex{T},1}) where T

    fill!(plan.plan1.upgrid[], zero(Complex{T}))
    interpolate!(plan.plan1, source, plan.plan1.upgrid[], :nu => :u)
    execute_type2!(plan.plan2, target, iflag, plan.plan1.upgrid[])

    target ./= plan.nuFkernel
end
