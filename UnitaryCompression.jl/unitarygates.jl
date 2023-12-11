mutable struct UnitaryGates
    dim::Int
    length::Int 
    depth::Int
    gates::Vector{Vector{Array{ComplexF64}}}
end

function UnitaryGates(dim::Int, length::Int, depth::Int; kwargs...)
    # Key arguments
    gatetype::String = get(kwargs, :gatetype, "random")
    rand::Real = get(kwargs, :rand, 0.1)

    # Create gates
    gates = [[] for i = 1:length-1]
    for i = 1:length-1
        num_gates = isodd(i) ? fld(depth + 1, 2) : fld(depth, 2)
        for _ = 1:num_gates
            push!(gates[i], gatetype == "random" ? random_gate(dim, rand) : identity_gate(dim))
        end
    end

    return UnitaryGates(dim, length, depth, gates)
end

function identity_gate(dim::Int)
    id = diagm(ones(ComplexF64, dim^2))
    id = reshape(id, (dim, dim, dim, dim))
    return id
end

function random_gate(dim::Int, delta=0.1)
    id = diagm(ones(ComplexF64, dim^2))
    id = reshape(id, (dim, dim, dim, dim))

    id .+= delta*(randn(Float64, dim, dim, dim, dim) .+ 1im .* randn(Float64, dim, dim, dim, dim))
    id, cmb = combineidxs(id, [3, 4])
    V1, S, V2 = svd(id, -1)
    id = contract(V1, V2, 3, 1) 
    id = uncombineidxs(id, cmb)
    return id
end

dim(gates::UnitaryGates) = gates.dim
length(gates::UnitaryGates) = gates.length

function make_unitary(gate::Array{ComplexF64})
    gate, cmb = combineidxs(gate, [3, 4])
    U, S, V = svd(gate, -1)
    gate = contract(U, V, 3, 1)
    gate = uncombineidxs(gate, cmb)
    return gate
end

function grow_size!(gates::UnitaryGates; kwargs...)
    # Key arguments
    gatetype::String = get(kwargs, :gatetype, "random")
    rand::Real = get(kwargs, :rand, 0.1)
    
    # Add more gates
    gates.length += 1
    gs = []
    num_gates = iseven(gates.length) ? fld(gates.depth + 1, 2) : fld(gates.depth, 2)
    for _ = 1:num_gates
        push!(gs, gatetype == "random" ? random_gate(gates.dim, rand) : identity_gate(gates.dim))
    end
    push!(gates.gates, gs)
    nothing
end

function grow_depth!(gates::UnitaryGates; kwargs...)
    # Key arguments
    gatetype::String = get(kwargs, :gatetype, "random")
    rand::Real = get(kwargs, :rand, 0.1)
    
    # Add more gates
    gates.depth += 1
    for i = 1:length(gates)-1
        if (iseven(i) && iseven(gates.depth)) || (isodd(i) && isodd(gates.depth))
            pushfirst!(gates.gates[i], gatetype == "random" ? random_gate(gates.dim, rand) : identity_gate(gates.dim))
        end
    end
    nothing
end

function GateList(gates::UnitaryGates)
    # Create the gates 
    sites = []
    gs = []
    for m = 1:gates.depth
        gs2 = []
        site = (isodd(m) ? 1 : 2)
        for i = site:2:gates.length-1
            push!(gs2, moveidx(gates.gates[i][end+1-fld(m+1, 2)], 2, 3))
            #push!(gs2, gates.gates[i][fld(m+1, 2)])
        end
        push!(gs, gs2)
        push!(sites, collect(site:2:gates.length-1))
    end
    return GateList(gates.length, sites, gs)
end

function MPO(st::Sitetypes, gates::UnitaryGates, cutoff=0)
    gatelist = GateList(gates)
    I = productMPO(st, ["id" for i = 1:gates.length])
    applygates!(I, gatelist; cutoff=cutoff)
    return I
end


### Save and write
function HDF5.write(parent::Union{HDF5.File, HDF5.Group}, name::AbstractString,
    gates::UnitaryGates)
    g = create_group(parent, name)
    attributes(g)["type"] = "UnitaryGates"
    attributes(g)["version"] = 1
    write(g, "dim", gates.dim)
    write(g, "length", gates.length)
    write(g, "depth", gates.depth)
    for i = 1:gates.length-1
        for j = 1:length(gates.gates[i])
            write(g, "gates[$(i)_$(j)]", gates.gates[i][j])
        end
    end
end


function HDF5.read(parent::Union{HDF5.File, HDF5.Group}, name::AbstractString,
    ::Type{UnitaryGates})
    g = open_group(parent, name)
    if read(attributes(g)["type"]) != "UnitaryGates"
        error("HDF5 group of file does not contain gates data.")
    end
    d = read(g, "dim")
    N = read(g, "length")
    M = read(g, "depth")
    gates = [[read(g, "gates[$(i)_$(j)]") for j=1:(isodd(i) ? fld(M+1, 2) : fld(M, 2))] for i=1:N-1]
    return UnitaryGates(d, N, M, gates)
end
