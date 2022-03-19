function next235even(n::Integer)
    if n <= 2
        return 2
    end
    (n % 2 == 1) ? n += 1 : nothing
    
    nplus = n - 2
    numdiv = 2

    while (numdiv > 1)
        nplus += 2 # search next even number
        numdiv = nplus

        while (numdiv % 2 == 0) 
            numdiv ÷= 2
        end

        while (numdiv % 3 == 0) 
            numdiv ÷= 3
        end

        while (numdiv % 5 == 0) 
            numdiv ÷= 5
        end
    end

    # even integer not less than N
    # with prime factors no larger than 5
    return nplus
end

function maxabs(x)
    maxabs = abs(maximum(x))
    minabs = abs(minimum(x))

    return (maxabs > minabs) ? maxabs : minabs
    # return abs(maximum(x) - minimum(x))/2
end

function arg_expr_helper!(argarray, D, argname, argtype)
    for d = 1:D
        push!(argarray, :($(Symbol(argname, d)) :: $(argtype)))
    end
end