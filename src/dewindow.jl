macro gen_dewindow(D, direction)
    #=
        direction > 0: type 1
        direction < 0: type 2
    =#
    local func_name = (direction > 0) ? :dewindow_type1! : :dewindow_type2!

    local args = []
    local contents = []

    #= 
        Function arguments
    =# 
    arg_expr_helper!(args, D, "N", :(Int))
    arg_expr_helper!(args, D, "Nup", :(Int))
    arg_expr_helper!(args, D, "Fkernel", :(CuDeviceArray{R,1}))
    push!( args, :($(:u) :: CuDeviceArray{C,$D}) )
    push!( args, :($(:u_pad) :: CuDeviceArray{C,$D}) )

    #= 
        Function contents
    =#
    local comm = :( idx = (blockIdx().x-1) * blockDim().x + threadIdx().x )
    push!(contents, comm)
    local comm = :( stride = blockDim().x * gridDim().x )
    push!(contents, comm)
    push!(contents, :(Ntotal = $(Expr(:call, :*, [Symbol("N",d) for d in 1:D]...))))
    
    local inside_loop = []

    # Julia is col-major; row is fastest, z is slowest
    # Index on (non-unsampled) uniform grid
    push!( inside_loop, :(i1 = (i-1) % N1) )
    if D > 1
        push!( inside_loop, :(i2 = ((i-1) ÷ N1) % N2) )
    end
    if D > 2
        push!( inside_loop, :(i3 = ((i-1) ÷ N1) ÷ N2) )
    end

    for d = 1:D # Periodic folding (periodic convolution)
        positive_shift = :($(Symbol("i",d)) - $(Symbol("N",d))÷2)
        negative_shift = :($(Symbol("i",d)) + $(Symbol("Nup",d)) - $(Symbol("N",d))÷2)
        push!( inside_loop, :($(Symbol("ip",d)) = #=
        =# ($(Symbol("i",d)) - $(Symbol("N",d))÷2 >= 0) ? $positive_shift : $negative_shift) )
    end

    # Julia is col-major; row is fastest, z is slowest
    if D == 1
        push!( inside_loop, :(uidx = i1 + 1) )
        push!( inside_loop, :(upadidx = ip1 + 1) )
    elseif D == 2
        push!( inside_loop, :(uidx = (i1+1) + N1*i2) )
        push!( inside_loop, :(upadidx = (ip1+1) + Nup1*ip2) )
    else
        push!( inside_loop, :(uidx = (i1+1) + N1*i2 + N1*N2*i3) )
        push!( inside_loop, :(upadidx = (ip1+1) + Nup1*ip2 + Nup1*Nup2*ip3) )
    end

    Fkernel = Expr(:call, :*, [:($(Symbol("Fkernel",d))[$(Symbol("i",d))+1]) for d in 1:D]...)
    if direction > 0
        push!( inside_loop, :(u[uidx] = u_pad[upadidx] / ($Fkernel)) )
    else
        push!( inside_loop, :(u_pad[upadidx] = u[uidx] / ($Fkernel)) )
    end

    inbounded = Expr(:macrocall, Symbol("@inbounds"), LineNumberNode(0), #=
    =# quote
        $(inside_loop...)
    end)

    push!(contents, Expr(:for, :(i = idx:stride:Ntotal), inbounded))
    push!(contents, :(return nothing))

    local func_def = quote
        function $(func_name)($(args...)) where {R<:FLT,C<:CPX}
            $(contents...)
        end
    end

    return esc(func_def)
end

# @macroexpand @gen_dewindow 3 -1
@gen_dewindow 1 1
@gen_dewindow 2 1
@gen_dewindow 3 1
@gen_dewindow 1 -1
@gen_dewindow 2 -1
@gen_dewindow 3 -1


function dewindow!(
    plan::Plan{D,R,C}, 
    u::CuArray{C,D}; 
    threads::Int=256) #=
    =# where {D,R<:FLT,C<:CPX}

    nftype = plan.nftype
    Nup = [plan.Nup1, plan.Nup2, plan.Nup3][1:D]
    N = [plan.N1, plan.N2, plan.N3][1:D]
    Fkernel = [plan.Fkernel1, plan.Fkernel2, plan.Fkernel3][1:D]
    Ntotal = *(N...)

    if nftype == 1
        @cuda threads=threads blocks=(Ntotal+threads-1)÷threads #=
        =# dewindow_type1!(
            N...,
            Nup...,
            Fkernel...,
            u,
            plan.upgrid
        )
    else
        @cuda threads=threads blocks=(Ntotal+threads-1)÷threads #=
        =# dewindow_type2!(
            N...,
            Nup...,
            Fkernel...,
            u,
            plan.upgrid
        )
    end
end