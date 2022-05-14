"""
References:
    - `base.atomics.jl` in Julia
    - https://github.com/tkf/ParallelIncrements.jl
"""

using Base: gc_alignment, llvmcall
using Base.Threads: ArithmeticTypes, WORD_SIZE, atomictypes, inttype, llvmtypes, FloatTypes

struct AtomicRef{T,A} <: Ref{T}
    pointer::Ptr{T}
    x::A
end
atomicref(x, args...) = AtomicRef(pointer(x, args...), x)


for typ in atomictypes
    lt = llvmtypes[typ]
    ilt = llvmtypes[inttype(typ)]
    rt = "$lt, $lt*"
    irt = "$ilt, $ilt*"
    @eval Base.getindex(ref::AtomicRef{$typ}) = let __x = ref.x; GC.@preserve __x begin
        llvmcall($"""
                 %ptr = inttoptr i$WORD_SIZE %0 to $lt*
                 %rv = load atomic $rt %ptr acquire, align $(gc_alignment(typ))
                 ret $lt %rv
                 """, $typ, Tuple{Ptr{$typ}}, ref.pointer)
    end end
    @eval Base.setindex!(ref::AtomicRef{$typ}, v::$typ) = let __x = ref.x; GC.@preserve __x begin
        llvmcall($"""
                 %ptr = inttoptr i$WORD_SIZE %0 to $lt*
                 store atomic $lt %1, $lt* %ptr release, align $(gc_alignment(typ))
                 ret void
                 """, Cvoid, Tuple{Ptr{$typ}, $typ}, ref.pointer, v)
    end end
    # Note: atomic_cas! succeeded (i.e. it stored "new") if and only if the result is "cmp"
    if typ <: Integer
        @eval Threads.atomic_cas!(ref::AtomicRef{$typ}, cmp::$typ, new::$typ) = let __x = ref.x; GC.@preserve __x begin
            llvmcall($"""
                     %ptr = inttoptr i$WORD_SIZE %0 to $lt*
                     %rs = cmpxchg $lt* %ptr, $lt %1, $lt %2 acq_rel acquire
                     %rv = extractvalue { $lt, i1 } %rs, 0
                     ret $lt %rv
                     """, $typ, Tuple{Ptr{$typ},$typ,$typ},
                     ref.pointer, cmp, new)
        end end
    else
        @eval Threads.atomic_cas!(ref::AtomicRef{$typ}, cmp::$typ, new::$typ) = let __x = ref.x; GC.@preserve __x begin
            llvmcall($"""
                     %iptr = inttoptr i$WORD_SIZE %0 to $ilt*
                     %icmp = bitcast $lt %1 to $ilt
                     %inew = bitcast $lt %2 to $ilt
                     %irs = cmpxchg $ilt* %iptr, $ilt %icmp, $ilt %inew acq_rel acquire
                     %irv = extractvalue { $ilt, i1 } %irs, 0
                     %rv = bitcast $ilt %irv to $lt
                     ret $lt %rv
                     """, $typ, Tuple{Ptr{$typ},$typ,$typ},
                     ref.pointer, cmp, new)
        end end
    end

    arithmetic_ops = [:add, :sub]
    for rmwop in [arithmetic_ops..., :xchg, :and, :nand, :or, :xor, :max, :min]
        rmw = string(rmwop)
        fn = Symbol("atomic_", rmw, "!")
        if (rmw == "max" || rmw == "min") && typ <: Unsigned
            # LLVM distinguishes signedness in the operation, not the integer type.
            rmw = "u" * rmw
        end
        if rmwop in arithmetic_ops && !(typ <: ArithmeticTypes) continue end
        if typ <: Integer
            @eval Threads.$fn(ref::AtomicRef{$typ}, v::$typ) = let __x = ref.x; GC.@preserve __x begin
                llvmcall($"""
                         %ptr = inttoptr i$WORD_SIZE %0 to $lt*
                         %rv = atomicrmw $rmw $lt* %ptr, $lt %1 acq_rel
                         ret $lt %rv
                         """, $typ, Tuple{Ptr{$typ}, $typ}, ref.pointer, v)
            end end
        else
            rmwop === :xchg || continue
            @eval Threads.$fn(ref::AtomicRef{$typ}, v::$typ) = let __x = ref.x; GC.@preserve __x begin
                llvmcall($"""
                         %iptr = inttoptr i$WORD_SIZE %0 to $ilt*
                         %ival = bitcast $lt %1 to $ilt
                         %irv = atomicrmw $rmw $ilt* %iptr, $ilt %ival acq_rel
                         %rv = bitcast $ilt %irv to $lt
                         ret $lt %rv
                         """, $typ, Tuple{Ptr{$typ}, $typ}, ref.pointer, v)
            end end
        end
    end
end

# Provide atomic floating-point operations via atomic_cas!
const opnames = Dict{Symbol, Symbol}(:+ => :add, :- => :sub)
for op in [:+, :-, :max, :min]
    opname = get(opnames, op, op)
    @eval function Threads.$(Symbol("atomic_", opname, "!"))(var::AtomicRef{T}, val::T) where T<:FloatTypes
        IT = inttype(T)
        old = var[]
        while true
            new = $op(old, val)
            cmp = old
            old = Threads.atomic_cas!(var, cmp, new)
            reinterpret(IT, old) == reinterpret(IT, cmp) && return old
            # Temporary solution before we have gc transition support in codegen.
            ccall(:jl_gc_safepoint, Cvoid, ())
        end
    end
end


"""
Currently, we do not have atomic_add for arbitrary array addresses in the standard library.       
`@floop` should be deactivated in `Pnufft.parallelize` for cpu operations 
if this atomic_add does not support multithreading.      
    
https://github.com/JuliaLang/julia/pull/37683      
https://discourse.julialang.org/t/atomic-operations-issue-on-staticarrays-with-cudanative/39576    
"""
function atomic_add!(arr::Union{Array,Base.ReinterpretArray}, idx::Integer, val)
    # prev = arr[idx]
    # arr[idx] += val
    # return prev

    r = atomicref(arr, idx)
    Threads.atomic_add!(r, val)
end
atomic_add!(arr::CuDeviceArray, idx::Integer, val) = CUDA.atomic_add!(pointer(arr, idx), val)