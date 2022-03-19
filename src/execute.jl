macro gen_execute_type3(D)

    local args = []
    local contents = []

    #= 
        Function arguments
    =# 
    push!( args, :($(:plan) :: Plan3{$D,R,C}) )
    push!( args, :($(:c) :: CuArray{C,1}) )
    push!( args, :($(:ftilde) :: CuArray{C,1}) )
    arg_expr_helper!(args, D, "x", :(CuArray{R,1}))
    arg_expr_helper!(args, D, "nu", :(CuArray{R,1}))

    #= 
        Function contents
    =#
    local sources = []
    local targets = []
    for d = 1:D
        push!(sources, :(length($(Symbol("x",d)))))
        push!(sources, :(==))
        push!(targets, :(length($(Symbol("nu",d)))))
        push!(targets, :(==))
    end
    push!(sources, :(length(c)))
    push!(sources, :(==))
    push!(sources, :(plan.plan1.Nnupts))
    push!(targets, :(length(ftilde)))
    push!(targets, :(==))
    push!(targets, :(plan.plan2.Nnupts))

    push!( contents, quote
        if safecheck
            $(Expr(:macrocall, Symbol("@assert"), LineNumberNode(0),
                Expr(:comparison, sources...),
                "Number of source points/coefficients mismatch"
            ))

            $(Expr(:macrocall, Symbol("@assert"), LineNumberNode(0),
                Expr(:comparison, targets...),
                "Number of target points/coefficients mismatch"
            ))
        end
    end)

    for d = 1:D
        push!( contents, :($(Symbol("x",d)) ./= plan.$(Symbol("gamma",d))) )
        # Here we additionally divide gamma1~3 by plan1.Nup1~3
        # to follow the definition of type 2 transform.
        push!( contents, :($(Symbol("nu",d)) .*= plan.$(Symbol("gamma",d)) / plan.plan1.$(Symbol("Nup",d))) )
    end

    local sources = Expr(:tuple, [Symbol("x",d) for d in 1:D]...)
    local targets = Expr(:tuple, [Symbol("nu",d) for d in 1:D]...)
    push!( contents, :(BinSort!(plan.plan1, ($sources)..., threads=threads)) )
    push!( contents, :(fill!(plan.plan1.upgrid, zero(C))) )
    push!( contents, :(interpolate!(plan.plan1, c, ($sources)..., threads=threads)) )
    push!( contents, :(execute!(plan.plan2, ftilde, plan.plan1.upgrid, ($targets)..., threads=threads, safecheck=safecheck)) )
    
    for d = 1:D
        push!( contents, :($(Symbol("x",d)) .*= plan.$(Symbol("gamma",d))) )
        push!( contents, :($(Symbol("nu",d)) ./= plan.$(Symbol("gamma",d)) / plan.plan1.$(Symbol("Nup",d))) ) 
    end

    push!( contents, :(ftilde ./= plan.nuFkernel) )
    

    local func_def = quote
        function execute!($(args...); threads::Int=256, safecheck::Bool=false) where {R<:FLT,C<:CPX}
            $(contents...)
        end
    end

    return esc(func_def)
end


function execute!(plan::Plan{D,R,C},
    c::CuArray{C,1}, ftilde::CuArray{C,D}, x::Vararg{CuArray{R,1},D}
    ;threads::Int=256, safecheck::Bool=false) where {D,R<:FLT,C<:CPX}

    if safecheck
        for xd in x
            @assert length(xd) == plan.Nnupts "Wrong number of nonuniform coordinates"
        end

        @assert length(c) == plan.Nnupts "Wrong number of nonuniform coefficients"
        @assert length(ftilde) == plan.N1*plan.N2*plan.N3 "Wrong number of uniform coefficients"
    end

    BinSort!(plan, x..., threads=threads)
    fill!(plan.upgrid, zero(C))
    if plan.nftype == 1
        interpolate!(plan, c, x..., threads=threads)
        plan.inplaceFFT*plan.upgrid
        dewindow!(plan, ftilde, threads=threads)
    else
        dewindow!(plan, ftilde, threads=threads)
        plan.inplaceFFT*plan.upgrid
        interpolate!(plan, c, x..., threads=threads)
    end
end

@gen_execute_type3 1
@gen_execute_type3 2
@gen_execute_type3 3


