#= 
    ProjGates calculates TN projections between the inner product of a 
    unitary BW circuit and an MPO.
=#

mutable struct ProjGates
    gates::UnitaryGates
    mpo::GMPS
    blocks::Vector{Array{Complex{Float64}}}
    center::Int
end


"""
    ProjGates(gates::UnitaryGates, mpo::GMPS)   

Create a tensor network projection between a brickwall circuit and and MPO.
"""
function ProjGates(gates::UnitaryGates, mpo::GMPS)
    projg = ProjGates(gates, mpo, [[] for i = 1:gates.length], 0)
    movecenter!(projg, length(gates))
    return projg
end

center(projg::ProjGates) = projg.center
length(projg::ProjGates) = projg.gates.length


"""
    block(projg::ProjGates, idx::Int)

Returns the block tensor at the given index.
"""
function block(projg::ProjGates, idx::Int)
    if idx <= 0
        return leftedgeblock(projg)
    elseif idx > length(projg.gates)
        return rightedgeblock(projg)
    else
        return projg.blocks[idx]
    end
end

"""
    leftedgeblock(projg::ProjGates)
    rightedgeblock(projg::ProjGates)

Returns the default edge blocks.
"""
function leftedgeblock(projg::ProjGates)
    prod = ones(ComplexF64, 1)
    num_gates = fld(projg.gates.depth, 2)
    id = diagm(ones(ComplexF64, projg.gates.dim))
    for _ = 1:num_gates
        prod = tensorproduct(prod, id)
    end
    return prod
end

function rightedgeblock(projg::ProjGates)
    prod = ones(ComplexF64, 1)
    num_gates = iseven(projg.gates.length) ? fld(projg.gates.depth, 2) : fld(projg.gates.depth+1, 2)
    id = diagm(ones(ComplexF64, projg.gates.dim))
    for _ = 1:num_gates
        prod = tensorproduct(prod, id)
    end
    return prod
end

"""
    movecenter!(projg::ProjGates, idx::Int)
    
Moves the canonical centre of the projection.
"""
function movecenter!(projg::ProjGates, idx::Int)
    if center(projg) == 0
        for i = 1:idx-1
            buildleft!(projg, i)
        end
        N = length(projg)
        for i = N:-1:idx+1
            buildright!(projg, i)
        end
    else
        if idx > center(projg)
            for i = center(projg):idx-1
                buildleft!(projg, i)
            end
        elseif idx < center(projg)
            for i = center(projg):-1:idx+1
                buildright!(projg, i)
            end
        end
    end
    projg.center = idx
    nothing
end


"""
    buildleft!(projg::ProjGates, idx::Int)

Builds the projection from the left.
"""
function buildleft!(projg::ProjGates, idx::Int)
    # Fetch objects
    left = block(projg, idx-1)
    mpo_block = moveidx(conj(projg.mpo[idx]), 3, 2)
    gates = idx < projg.gates.length ? projg.gates.gates[idx] : []

    # Contract with MPO 
    if isodd(projg.gates.depth) && isodd(idx)
        left = contract(mpo_block, left, 1, 1)
        left = moveidx(left, 1, length(size(left)))
        left = moveidx(left, 1, 2)
    elseif isodd(projg.gates.depth) && iseven(idx)
        left = contract(mpo_block, left, [1, 2, 3], [1, length(size(left)), 2])
    elseif iseven(projg.gates.depth) && isodd(idx)
        left = contract(mpo_block, left, [1, 3], [1, 2])
        left = moveidx(left, 1, length(size(left)))
    else
        left = contract(mpo_block, left, [1, 2], [1, length(size(left))])
        left = moveidx(left, 1, 2)
    end

    # Contract with gates 
    for i = 1:length(gates)
        left = contract(left, gates[i], [2, 3], [1, 3])
    end
    
    """
    # Trace if at end 
    if idx == length(projg)
        while length(size(left)) > 1
            left = trace(left, 2, 3)
        end
    end
    """

    projg.blocks[idx] = left
    nothing
end

"""
    buildright!(projg::ProjGates, idx::Int)

Builds the projection from the right.
"""
function buildright!(projg::ProjGates, idx::Int)
    # Fetch objects
    right = block(projg, idx+1)
    mpo_block = moveidx(conj(projg.mpo[idx]), 3, 2)
    gates = idx < projg.gates.length ? projg.gates.gates[idx] : []

    # Contract with gates 
    for i = 1:length(gates)
        right = contract(gates[length(gates)+1-i], right, [2, 4], [length(size(right))-1, length(size(right))])
    end

    if idx == length(projg)
        right = moveidx(right, 1, length(size(right)))
    end

    # Contract with MPO 
    if isodd(projg.gates.depth) && isodd(idx)
        right = contract(mpo_block, right, [4, 2, 3], [length(size(right)), length(size(right))-1, 1])
    elseif isodd(projg.gates.depth) && iseven(idx)
        right = contract(mpo_block, right, 4, length(size(right)))
        right = moveidx(right, 2, length(size(right)))
    elseif iseven(projg.gates.depth) && isodd(idx)
        right = contract(mpo_block, right, [4, 2], [length(size(right)), length(size(right))-1])
    else
        right = contract(mpo_block, right, [4, 3], [length(size(right)), 1])
        right = moveidx(right, 2, length(size(right)))
    end
    
    """
    # Trace if at end 
    if idx == 1
        while length(size(right)) > 1
            right = trace(right, 2, 3)
        end
    end
    """

    projg.blocks[idx] = right
    nothing
end


"""
    projectgate(projg::ProjGates, idx::Int, which::Int)

Determine the projection of the TN onto a single gate.
"""
function projectgate(projg::ProjGates, idx::Int, which::Int)
    # Fetch objects
    left = block(projg, idx-1)
    right = block(projg, idx+1)
    mpo_block = moveidx(conj(projg.mpo[idx]), 3, 2)
    gates = idx < projg.gates.length ? projg.gates.gates[idx] : []

    # Contract with MPO 
    if isodd(projg.gates.depth) && isodd(idx)
        left = contract(mpo_block, left, 1, 1)
        left = moveidx(left, 1, length(size(left)))
        left = moveidx(left, 1, 2)
    elseif isodd(projg.gates.depth) && iseven(idx)
        left = contract(mpo_block, left, [1, 2, 3], [1, length(size(left)), 2])
    elseif iseven(projg.gates.depth) && isodd(idx)
        left = contract(mpo_block, left, [1, 3], [1, 2])
        left = moveidx(left, 1, length(size(left)))
    else
        left = contract(mpo_block, left, [1, 2], [1, length(size(left))])
        left = moveidx(left, 1, 2)
    end

    # Contract with gates 
    for i = 1:length(gates)
        if i != which
            left = contract(left, gates[i], i < which ? [2, 3] : [4, 5], [1, 3])
        end
    end
    left = moveidx(left, 1, 3)

    # contract with right 
    right = moveidx(right, 2 + 2*(which-1), 1)
    right = moveidx(right, 3 + 2*(which-1), 2)
    left = contract(left, right, collect(3:length(size(left))), collect(3:length(size(right))))
    left = moveidx(left, 2, 3)

    return left
end

"""
    inner(gates::UnitaryGates, U)

Calculate the inner product of the TN.
"""
function inner(gates::UnitaryGates, U)
    projg = ProjGates(gates, U)
    buildleft!(projg, length(gates))
    left = block(projg, length(gates))
    right = block(projg, length(gates)+1)
    return contract(left, right, collect(1:length(size(left))), collect(1:length(size(right))))[]
end