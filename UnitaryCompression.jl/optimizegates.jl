"""
    OptimizeGates(gates::UnitaryGates, U::GMPS; kwargs...)

Variationally optimize a brickwall unitary circuit to maximize the overlap
with an MPO, U.
"""
function OptimizeGates(gates::UnitaryGates, U::GMPS; kwargs...)
    # Convergence criteria
    minsweeps::Int = get(kwargs, :minsweeps, 1)
    maxsweeps::Int = get(kwargs, :maxsweeps, 10^5)
    maxvsweeps::Int = get(kwargs, :maxvsweeps, 100)
    tol::Float64 = get(kwargs, :tol, 1e-5)
    vtol::Float64 = get(kwargs, :vtol, tol * 1e-1)
    numconverges::Float64 = get(kwargs, :numconverges, 3)
    verbose::Bool = get(kwargs, :verbose, 1)

    # Gradient decent
    epsilon::Float64 = get(kwargs, :epsilon, 0.0)

    # Create the projection 
    projg = ProjGates(gates, U)
    movecenter!(projg, length(gates))
    buildright!(projg, length(gates))

    # Calculate the cost 
    cost_mpo = real(trace(adjoint(U), U))
    function calculate_cost()
        if projg.center > 1
            buildleft!(projg, projg.center-1)
        end
        buildright!(projg, projg.center)
        left = block(projg, projg.center-1)
        right = block(projg, projg.center)
        cost = contract(left, right, collect(1:length(size(left))), collect(1:length(size(right))))[]
        return sqrt(abs(2.0^(length(gates)) + cost_mpo - 2*real(cost)) / 2.0^(length(gates)))
    end

    #  Loop through until convergence
    direction = true
    converged = false
    convergedsweeps = 0
    sweeps = 0
    lastcost = calculate_cost()
    cost = calculate_cost()
    diff(x, y) = abs(x) < 1e-10 ? x-y : 2*(x - y) / (x+y)
    costs = [cost]
    while !converged
        #println("--------")
        for j = 1:length(gates)-2
            # Determine the site
            site = direction ? length(gates) - j : 1 + j

            # Build the projector blocks
            movecenter!(projg, site)
            buildright!(projg, site)

            converged2 = false
            direction2 = true
            lastcost2 = deepcopy(cost) 
            vsweeps = 0
            while !converged2
                # Update gates
                whichs = direction2 ? collect(1:gates.depth) : collect(gates.depth:-1:1)
                for i = whichs
                    # Find which gate 
                    if (isodd(site) && isodd(gates.depth)) || (iseven(site) && iseven(gates.depth))
                        site2 = site - ((i-1) % 2)
                    else
                        site2 = site - 1 + ((i-1) % 2)
                    end
                    which = 1 + fld(i-1, 2)

                    if 0 < site2 < gates.length
                        gate = make_unitary(conj(projectgate(projg, site2, which)))
                        if epsilon != 0
                            gate = make_unitary(epsilon * gates.gates[site2][which] + (1-epsilon) * gate)
                        end
                        gates.gates[site2][which] = gate 
                        if site2 < site
                            buildleft!(projg, site2)
                        else
                            buildright!(projg, site2)
                        end
                    end
                end

                # Check convergence
                direction2 = !direction2
                cost = calculate_cost()
                vsweeps += 1
                if diff(lastcost2, cost) < vtol
                    converged2 = true
                end
                if vsweeps >= maxvsweeps && maxvsweeps != 0
                    converged2 = true
                end
                lastcost2 = copy(cost)
            end
        end

        # Reverse direction
        direction = !direction

        # Check convergence
        push!(costs, cost)
        sweeps += 1
        if sweeps >= minsweeps
            if diff(lastcost, cost) < tol
                convergedsweeps += 1
            else
                convergedsweeps = 0
            end
            if convergedsweeps >= numconverges
                converged = true
            end
            if sweeps >= maxsweeps && maxsweeps != 0
                converged = true
            end
        end
        lastcost = copy(cost)

        # Output information
        if verbose
            @printf("Sweep=%d, cost=%.12f \n", sweeps, real(cost))
        end
    end

    return costs
end
