"""
direction > 0: NU_to_U
direction < 0: U_to_NU
"""
macro gen_dewindow(D, direction, gpu)
    local func_name = (direction > 0) ? :dewindow_type1! : :dewindow_type2!

    # Julia is col-major; row is fastest, z is slowest
    # Index on (non-unsampled) uniform grid
    local ugrid_cart_index = [
        :(i1 = (i-1) % N1),
        :(i2 = ((i-1) ÷ N1) % N2),
        :(i3 = ((i-1) ÷ N1) ÷ N2)
    ][1:D]

    # Periodic folding (periodic convolution)
    local pbc = []
    for d = 1:D 
        i_d = var(:i,d)
        ip_d = var(:ip,d)
        N_d = var(:N,d)
        Nup_d = var(:Nup,d)

        push!(pbc, :($ip_d = $i_d - $N_d÷2))
        push!(pbc, :($ip_d = $ip_d + ($ip_d >= 0 ? 0 : $Nup_d)))
    end

    # Julia is col-major; row is fastest, z is slowest
    local u_scalar_index = [
        :(uidx = i1 + 1),
        :(upadidx = ip1 + 1),
        :(uidx = (i1+1) + N1*i2),
        :(upadidx = (ip1+1) + Nup1*ip2),
        :(uidx = (i1+1) + N1*i2 + N1*N2*i3),
        :(upadidx = (ip1+1) + Nup1*ip2 + Nup1*Nup2*ip3)
    ][(2*D-1):(2*D)]

    Fkernel = Expr(:call, :*, [:($(var(:Fkernel,d))[$(var(:i,d))+1]) for d in 1:D]...)
    dewindow = direction > 0 ? #=
    =# :(u[uidx] = u_pad[upadidx] / ($Fkernel)) : #=
    =# :(u_pad[upadidx] = u[uidx] / ($Fkernel))

    u = gpu ? :u : :(u::Array)
    local func_def = quote
        function $(func_name)(Ntotal, $(argGen(:N, 1:D)...), $(argGen(:Nup, 1:D)...), 
            $(argGen(:Fkernel, 1:D)...), u_pad, $u)
            
            $(parallelize(
                quote
                    $(ugrid_cart_index...)
                    $(pbc...)
                    $(u_scalar_index...)
                    $dewindow
                end,
                :Ntotal, gpu)...)

            return nothing
        end
    end

    return esc(func_def)
end


# @macroexpand @gen_dewindow 3 -1
for d = (1,2,3)
    for direction = (1,-1)
        for gpu = (true, false)
            @eval @gen_dewindow $d $direction $gpu
        end
    end
end


function dewindow!(plan::NuFFTPlan{T,D,FK}, upgrid, u, nftype::Pair{Symbol,Symbol}) where {T,D,FK} 
    auto_config = FK <: CuArray ? auto_config_gpu : auto_config_cpu

    Ntotal = *(plan.N...)

    func = nftype == (:nu => :u) ? dewindow_type1! : dewindow_type2!
    auto_config(
        Ntotal,
        func,
        Ntotal,
        plan.N...,
        plan.Nup...,
        plan.Fkernel...,
        upgrid,
        u
    )
end
