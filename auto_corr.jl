using StatsBase
using Plots
using Base.Threads
include("simulation.jl")

L::Int64 = 50 # Lattice size
J::Float64 = 1.0 # Interaction strength
k::Float64 = 1.0 # Boltzmann constant 
Tc::Float64 = 2 / (log(1 + sqrt(2))) * J / k # Critical temperature
Tmin::Float64, Tmax::Float64 = ((Tc - 1)*k, (Tc + 1)*k)

T_list::Vector{Float64} = [Tmin, Tc, Tmax]
monte_steps::Int64 = 2000
eq_steps::Int64 = 3000

# Initialize lattice

E = zeros(length(T_list), monte_steps)
M = zeros(length(T_list), monte_steps)

rngs = [MersenneTwister() for _ in 1:Threads.nthreads()]
@threads for i in eachindex(T_list)
    rng = rngs[Threads.threadid()]
    lattice = rand(rng, [-1, 1], L, L)
    for j in 1:eq_steps
        lattice = metropolis_step(lattice, T_list[i], J, false)
    end
    for j in 1:monte_steps
        lattice = metropolis_step(lattice, T_list[i], J, false)
        E[i, j] = energy(lattice, J)
        M[i, j] = magnetization(lattice)
    end
end

num_lags = 0:200
E_corr::Matrix{Float64} = zeros(Float64, length(T_list), length(num_lags))
M_corr::Matrix{Float64} = zeros(Float64, length(T_list), length(num_lags))
for i in eachindex(T_list)
    E_corr[i, :] = autocor(E[i, :], num_lags)
    M_corr[i, :] = autocor(M[i, :], num_lags)
end

p1 = plot(num_lags, M_corr[1, :], label="T < Tc", xlabel="Lag", ylabel="acf(M)", title="Energy autocorrelation vs lag")
plot!(p1, num_lags, M_corr[2, :], label="T = Tc")
plot!(p1, num_lags, M_corr[3, :], label="T > Tc")

p2 = plot(num_lags, E_corr[1, :], label="T < Tc", xlabel="Lag", ylabel="acf(E)", title="Energy autocorrelation vs lag")
plot!(p2, num_lags, E_corr[2, :], label="T = Tc")
plot!(p2, num_lags, E_corr[3, :], label="T > Tc")

plot(p1, p2, layout=(2, 1), size=(600, 700))

savefig("figs/autocorrelation.pdf")
closeall()


