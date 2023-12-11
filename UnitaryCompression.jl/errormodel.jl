"""
    randomHermitian(dim::Int)

Create a random Hermitian matrix with dimensions dim.
"""
function randomHermitian(dim::Int)
    H = zeros(ComplexF64, dim, dim)
    for i = 1:dim
        for j = i:dim
            H[i, j] = randn(ComplexF64)
            H[j, i] = conj(H[i, j])
        end
        H[i, i] = real(H[i, i])
    end
    return H
end

"""
    randomUnitary(dim::Int, weight::Real = 1.0)

Create a random unitary matrix with dimensions dim.
The strength of the randomness is given by weight.
"""
function randomUnitary(dim::Int, weight::Real = 1.0)
    return exp(-1im*weight*randomHermitian(dim))
end

function randomBrickwall(length::Int, depth::Int)
    # Create the gates 
    sites = []
    gs = []
    for m = 1:depth
        gs2 = []
        site = (isodd(m) ? 1 : 2)
        for _ = site:2:length-1
            gate = moveidx(reshape(randomUnitary(4), (2, 2, 2, 2)), 2, 3)
            push!(gs2, gate)
        end
        push!(gs, gs2)
        push!(sites, collect(site:2:length-1))
    end
    return GateList(length, sites, gs)
end


"""
    addnoise!(gates::GateList, weight::Real=0.1)

Make a gate list noisy; the strenth of the noise is weight.
"""
function addnoise!(gates::GateList, weight::Real=0.1)
    for m = 1:length(gates.gates)
        for j = 1:length(gates.gates[m])
            # Fetch gate and dimensions
            gate = gates.gates[m][j]
            dims = size(gate)
            dim = prod(dims[1:2:length(dims)])

            # Make a random gate
            randgate = randomUnitary(dim, weight)
            randgate = reshape(randgate, dims[1:2:length(dims)]..., dims[2:2:length(dims)]...)
            
            # Apply the gate
            gate = contract(randgate, gate, collect(Int(length(dims)/2)+1:length(dims)), collect(1:2:length(dims)))
            for i = 1:Int(length(dims)/2)
                gate = moveidx(gate, Int(length(dims)/2)+i, 2*i)
            end

            gates.gates[m][j] = gate
        end
    end

    return nothing
end