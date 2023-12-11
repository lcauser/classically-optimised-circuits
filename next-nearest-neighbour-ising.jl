include("TensorNetworks.jl/TensorNetworks.jl")
include("UnitaryCompression.jl/UnitaryCompression.jl")

### Parameters
N = 8 # System Size 
gx = 1.0
gzz = 0.0
gz1z = 1.0
t = 1e-1 # Time step
depth = 6 # Circuit depth 
cutoff = 1e-16 # Variational cutoff for MPOs 


### Create the Hamiltonian 
sh = spinhalf()
function hamiltonian(N, gx, gzz, gz1z)
    # Create H 
    H = OpList(N)
    for i = 1:N
        add!(H, "x", i, gx)
    end
    for i = 1:N-1
        add!(H, ["z", "z"], [i, i+1], gzz)
    end
    for i = 1:N-2
        add!(H, ["z", "id", "z"], [i, i+1, i+2], gz1z)
    end

    return H
end
H = hamiltonian(N, gx, gzz, gz1z)

### Make an initial guess
# SWAP Gates
swap = zeros(ComplexF64, 2, 2, 2, 2)
swap[1, 1, 1, 1] = 1
swap[2, 2, 2, 2] = 1
swap[1, 2, 2, 1] = 1
swap[2, 1, 1, 2] = 1

# Make a circuit of SWAPs and identity matrices
gates = UnitaryGates(2, N, depth; rand=0.0)
for i = 1:4:N-1
    gates.gates[i][end] = deepcopy(swap)
end
for i = 1:2:N-1
    gates.gates[i][end-1] = deepcopy(swap)
end
for i = 3:4:N-1
    gates.gates[i][end-2] = deepcopy(swap)
end

### Optimize the circuit
# Create U and and optimize
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