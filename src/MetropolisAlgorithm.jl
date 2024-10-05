module MetropolisAlgorithm

export metropolis_step, run_simulation

using Random
using Base.Threads
include("../src/Utils.jl")
using .Utils: energy, magnetization

function metropolis_step(lattice::Array{<:Integer,2}, T::Real, J::Real,
    parallel::Bool=false)::Array{Int64,2}

    L = size(lattice, 1)
    if parallel
        rngs = [MersenneTwister() for _ in 1:Threads.nthreads()]
        Threads.@threads for _ in 1:L^2
            # Use thread-local RNG
            rng = rngs[Threads.threadid()]
            i, j = rand(rng, 1:L, 2)
            dE = 2J * lattice[i, j] *
                 (
                     lattice[mod1(i + 1, L), j] + lattice[mod1(i - 1, L), j] +
                     lattice[i, mod1(j + 1, L)] + lattice[i, mod1(j - 1, L)]
                 )

            if dE < 0 || rand(rng) < exp(-dE / T)
                # Modification is inherently thread-safe assuminging if no two threads
                # modify the same cell at the same time.
                lattice[i, j] *= -1
            end
        end
    else
        rng = Random.GLOBAL_RNG
        for _ in 1:L^2
            i, j = rand(rng, 1:L, 2)
            dE = 2J * lattice[i, j] *
                 (
                     lattice[mod1(i + 1, L), j] + lattice[mod1(i - 1, L), j] +
                     lattice[i, mod1(j + 1, L)] + lattice[i, mod1(j - 1, L)]
                 )

            if dE < 0 || rand(rng) < exp(-dE / T)
                lattice[i, j] *= -1
            end
        end
    end
    return lattice
end

function run_simulation(
    lattice::Array{<:Integer,2},
    T::Real,
    J::Real,
    annealing::Bool=false,
    monte_steps::Integer=1000,
    eq_steps::Integer=2000)::Tuple{Float64,Float64,Float64,Float64}

    E_avg::Float64 = 0.0
    M_avg::Float64 = 0.0
    Cv::Float64 = 0.0
    χ::Float64 = 0.0
    if annealing
        T_ann = collect(range(3.0, T, length=eq_steps))
        for t in T_ann
            lattice = metropolis_step(lattice, t, J)
        end
        for _ in 1:monte_steps
            lattice = metropolis_step(lattice, T, J)
            E = energy(lattice, J)
            M = magnetization(lattice)
            E_avg += E
            M_avg += M
            Cv += E^2
            χ += M^2
        end
        E_avg /= monte_steps
        M_avg /= monte_steps
        Cv = (Cv / monte_steps - E_avg^2) / (T^2)
        χ = (χ / monte_steps - M_avg^2) / T
    else
        for _ in 1:monte_steps
            lattice = metropolis_step(lattice, T, J)
            E = energy(lattice, J)
            M = magnetization(lattice)
            E_avg += E
            M_avg += M
            Cv += E^2
            χ += M^2
        end
        E_avg /= monte_steps
        M_avg /= monte_steps
        Cv = (Cv / monte_steps - E_avg^2) / (T^2)
        χ = (χ / monte_steps - M_avg^2) / T
    end
    return E_avg, M_avg, Cv, χ
end
end