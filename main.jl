include("src/MetropolisAlgorithm.jl")
using .MetropolisAlgorithm: run_simulation

using Base.Threads, ProgressMeter
using Plots, LaTeXStrings

L::Int64 = 30 # Lattice size
J::Float64 = 1.0 # Interaction strength
k::Float64 = 1.0 # Boltzmann constant 
Tc::Float64 = 2 / (log(1 + sqrt(2))) * J / k # Critical temperature
Tmin::Float64, Tmax::Float64 = (Tc / 100 * k, Tc * 2 * k)

T_list::Vector{Float64} = collect(range(Tmin, Tmax, length=100))
Cv::Vector{Float64} = zeros(Float64, length(T_list))
M_avg::Vector{Float64} = zeros(Float64, length(T_list))
E_avg::Vector{Float64} = zeros(Float64, length(T_list))
Chi::Vector{Float64} = zeros(Float64, length(T_list))

println("Running simulation... \n")

@showprogress @threads for i in eachindex(T_list)
    if T_list[i] < Tc
        E_avg[i], M_avg[i], Cv[i], Chi[i] = run_simulation(rand([1, -1], L, L), T_list[i], J, true)
    else
        E_avg[i], M_avg[i], Cv[i], Chi[i] = run_simulation(rand([1, -1], L, L), T_list[i], J, false)
    end
end

println("Simulation finished!")


# Create each subplot
p1 = scatter(T_list / Tc, E_avg / L^2,
    label="Energy", color="blue", marker=:circle,
    alpha=0.4, markersize=2.0, xlabel=L"T/ T_c",
    ylabel=L"\langle E \rangle / N^2", legend=:topright, grid=true)
p2 = scatter(T_list / Tc, M_avg / L^2,
    label="Magnetization", color="red", marker=:circle,
    alpha=0.4, markersize=2.0, xlabel=L"T/ T_c",
    ylabel=L"\langle M \rangle / N^2", legend=:topright, grid=true)
p3 = scatter(T_list / Tc, Cv / L^2,
    label="Specific Heat", color="green", marker=:circle,
    alpha=0.4, markersize=2.0, xlabel=L"T/ T_c",
    ylabel=L"\frac{C_v}{k_B^2  N^2}", legend=:topright, grid=true)
p4 = scatter(T_list / Tc, Chi / L^2,
    label="Susceptibility", color="purple", marker=:circle,
    alpha=0.4, markersize=2.0, xlabel=L"T/ T_c",
    ylabel=L"\frac{\chi}{k_B  N^2}", legend=:topright, grid=true)


# Combine subplots into a 2x2 grid layout
plot(p1, p2, p3, p4, layout=(2, 2), size=(1000, 800),)
plot!(title=L"N = " * "$L, " * L"J = " * "$J", titlefontsize=12)
# Save the figure
savefig("figs/ising_model_plots.pdf")
closeall()

