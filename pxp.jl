include("TensorNetworks.jl/TensorNetworks.jl")
include("UnitaryCompression.jl/UnitaryCompression.jl")

### Parameters
N = 14 # System Size 
t = 1e-1 # Time step
depth = 4 # Circuit depth 
cutoff = 1e-16 # Variational cutoff for MPOs 


### Create the Hamiltonian 
sh = spinhalf()
function hamiltonian(N)
    # Create H 
    H = OpList(N)
    for i = 1:N-2
        add!(H, ["pd", "x", "pd"], [i, i+1, i+2], 1)
    end
    add!(H, ["x", "pd"], [1, 2], 1)
    add!(H, ["pd", "x"], [N-1, N], 1)

    return H
end
H = hamiltonian(N)

### Optimize the circuit
gates = UnitaryGates(2, N, depth) # Create a BW circit

# Optimize by annealing time
dts = 10 .^ (0:-0.1:-2)
costs = []
for dt in dts 
    dt <= t && break

    # Create MPO to optimize against
    if N <= 12
        U = exp_mpo(sh, H, dt; cutoff=cutoff)
    else
        U = trotter_mpo(sh, H, dt, 100; cutoff=cutoff)
    end
    cost = OptimizeGates(gates, U; tol=1e-5, maxsweeps=10000, maxvsweeps=1)

end

# Final step
if N <= 12
    U = exp_mpo(sh, H, t; cutoff=cutoff)
else
    U = trotter_mpo(sh, H, t, 100; cutoff=cutoff)
end
cost = OptimizeGates(gates, U; tol=1e-6, maxsweeps=10000, maxvsweeps=1)

### Measure observables
gates_mpo = MPO(sh, gates, cutoff) # Turn gates into MPO

# Calculate the error density 
error_density =  sqrt(2 - real(trace(gates_mpo, adjoint(U))) ^ (1/N))

# Calculate the error in energy conservation 
H = MPO(sh, H) # Turn the Hamiltonian into an MPO 
error_energy = sqrt(real(2*(trace(H, H) - trace(H, gates_mpo, H, adjoint(gates_mpo))) / trace(H, H)))
  
# Calculate infidelities
infidelity, infidelity_local = 1 .- calc_fidelity(U, gates_mpo)