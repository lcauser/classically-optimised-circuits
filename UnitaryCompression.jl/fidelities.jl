function calc_fidelity(Umpo::GMPS, Ubw::GMPS)
    # Loop through to calculate each
    N = length(Ubw)
    fidelities = []
    for j = 0:N
        left = ones(ComplexF64, 1, 1, 1, 1)
        for i = 1:N
            left = contract(left, conj(Ubw[i]), 1, 1)
            left = contract(left, Umpo[i], [1, 5], [1, 2])
            left = contract(left, conj(Umpo[i]), 1, 1)
            left = contract(left, Ubw[i], [1, 7], [1, 2])
            if i == j || j == 0
                delta = zeros(ComplexF64, 2, 2, 2, 2)
                delta[1, 1, 1, 1] = delta[2, 2, 2, 2] = 1
                left = contract(left, delta, [1, 3, 5, 7], [1, 2, 3, 4])
            else
                left = trace(left, 1, 7)
                left = trace(left, 2, 4)
            end
        end
        push!(fidelities, left[])
    end

    # Calculate fideliteis
    F = real(fidelities[1]) / 2.0^(N)
    Flocal = real(1 / (N * 2.0^(N)) * sum(fidelities[2:end]))
    return F, Flocal
end