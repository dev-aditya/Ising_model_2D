function energy(
    lattice::Array{Int64,2},
    J::Float64)::Float64

    E::Float64 = 0.0
    L = size(lattice, 1)
    for i in 1:L, j in 1:L
        E += -J * lattice[i, j] *
             (
                 lattice[mod1(i + 1, L), j] + lattice[mod1(i - 1, L), j] +
                 lattice[i, mod1(j + 1, L)] + lattice[i, mod1(j - 1, L)]
             )
    end
    return E / 2
end

function magnetization(lattice::Array{Int64,2})::Float64
    return sum(lattice)
end