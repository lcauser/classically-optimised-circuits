"""
    exp_matrix(st::Sitetypes, ops::OpList, dt::Number = 1im)

Create an exponential matrix from an operator list.
"""
function exp_matrix(st::Sitetypes, ops::OpList, dt::Number = 1im)
    # Create the Hamiltonian
    H = zeros(ComplexF64, (st.dim ^ ops.length, st.dim ^ ops.length))
    for i = 1:length(ops.ops)
        oper = 1
        for site = 1:ops.length
            if site in ops.sites[i]
                idx = findfirst(site .== ops.sites[i])
                oper = kron(oper, op(st, ops.ops[i][idx]))
            else
                oper = kron(oper, op(st, "id"))
            end
        end
        H += ops.coeffs[i] * oper
    end

    # Exponentiate
    return exp(-1im * dt * H)
end

"""
    exp_mpo(st::Sitetypes, ops::OpList, dt::Number = 1im; kwargs...)

Create an exponential MPO from an operator list.
"""
function exp_mpo(st::Sitetypes, ops::OpList, dt::Number = 1im; kwargs...)
    # Create the matrix
    V = exp_matrix(st, ops, dt)
    V = reshape(V, (1, size(V)...))

    # Decompose into MPO 
    U_mpo = MPO(st.dim, ops.length)
    for site = 1:ops.length-1
        V = reshape(V, (size(V)[1], st.dim, Int(size(V)[2] / st.dim), st.dim, Int(size(V)[2] / st.dim)))
        V, cmb = combineidxs(V, [3, 5])
        U, S, V = svd(V, -1; kwargs...)
        V = contract(S, V, 2, 1)
        V = reshape(V, (size(V)[1], cmb[2][1], cmb[2][2]))
        U_mpo[site] = U
    end
    V = reshape(V, (size(V)..., 1))
    U_mpo[ops.length] = V
    return U_mpo
end

"""
    trotter_mpo(st::Sitetypes, H::OpList, dt::Number, steps::Int; kwargs...)

Create an MPO approximation of a time-evolution operator using Trotterization.
"""
function trotter_mpo(st::Sitetypes, H::OpList, dt::Number, steps::Int; kwargs...)
    # Create the gates
    gates = trotterize(st, -1im*H, dt / steps)
    U = productMPO(st, ["id" for i = 1:H.length])

    # Evolve
    for _ = 1:steps
        applygates!(U, gates; kwargs...)
    end

    return U
end

"""
    trotter1(st::Sitetypes, H::OpList, dt::Number; kwargs...)

Create a first-order Trotter MPO.
"""
function trotter1(st::Sitetypes, H::OpList, dt::Number; kwargs...)
    # Split the operator list into seperate lists 
    ops = [OpList(H.length) for _ = 1:siterange(H)]
    for i = 1:length(H.ops)
        rng = maximum(H.sites[i]) - minimum(H.sites[i]) + 1
        add!(ops[rng], H.ops[i], H.sites[i], H.coeffs[i])
    end

    # Trotterize gates 
    gates = [trotterize(st, -1im*ops[i], dt; order=1) for i = 1:siterange(H)]

    # Create MPO
    U = productMPO(st, ["id" for i = 1:H.length])
    for i = 1:siterange(H)
        applygates!(U, gates[i]; kwargs...)
    end

    return U
end

"""
    trotter2(st::Sitetypes, H::OpList, dt::Number; kwargs...)

Create a second-order Trotter MPO.
"""
function trotter2(st::Sitetypes, H::OpList, dt::Number; kwargs...)
    # Split the operator list into seperate lists 
    ops = [OpList(H.length) for _ = 1:siterange(H)]
    for i = 1:length(H.ops)
        rng = maximum(H.sites[i]) - minimum(H.sites[i]) + 1
        add!(ops[rng], H.ops[i], H.sites[i], H.coeffs[i])
    end

    # Trotterize gates 
    gates = [trotterize(st, -1im*ops[i], i == siterange(H) ? dt : (dt/2); order = i == siterange(H) ? 2 : 1) for i = 1:siterange(H)]

    # Create MPO
    U = productMPO(st, ["id" for i = 1:H.length])
    for i = 1:siterange(H)
        applygates!(U, gates[i]; kwargs...)
    end
    for i = siterange(H)-1:-1:1
        applygates!(U, gates[i]; kwargs...)
    end

    return U
end

"""
    trace_mpo(U::GMPS, sites::Int; kwargs...)

Partially trace an MPO.
"""
function trace_mpo(U::GMPS, sites::Int; kwargs...)
    A = U[length(U)]
    for i = 1:length(U)-sites 
        A = trace(A, 2, 3)
        A = contract(U[length(U)-i], A, 4, 1)
    end

    U2 = MPO(U.dim, sites)
    U2.tensors[1:sites-1] = deepcopy(U.tensors[1:sites-1])
    U2[sites] = A 
    movecenter!(U2, sites)
    movecenter!(U2, 1; kwargs...)

    return U2
end