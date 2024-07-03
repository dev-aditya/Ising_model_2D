using Plots
using ProgressMeter
include("simulation.jl")
include("utilities.jl")

# Parameters 
N_grid::Int64 = 100  # Size of the lattice
J::Float64 = 1.0       # Interaction strength
N_steps::Int64 = 1000  # Number of simulation steps
Tc::Float64 = 2 * J / log(1 + sqrt(2)) # Critical temperature
T::Float64 = 0.9  # Temperature

# Initialize the lattice
lattice = rand([-1, 1], N_grid, N_grid)

# Function to plot the lattice
function plot_grid(lattice, step, T)
    N_grid = size(lattice, 1)
    m = round(magnetization(lattice) / N_grid^2, digits=2)
    E = round( energy(lattice, J) / N_grid^2, digits=2)
    heatmap(1:N_grid, 1:N_grid, lattice,
        aspect_ratio=:equal, c=:grays, yflip=true,
        xlabel="N = $N_grid", ylabel="N = $N_grid",
        legend=true, border=:none, ticks=false)
    title!("Step: $step, \n T = $T, m = $(m), E = $(E)")
end
p = Progress(N_steps, 1, "Simulation Progress: ", 50)
if T <= Tc 
    println("Starting Simulation with Annealing ...")
    T_ann = zeros(1000)
    n_ann = Int64(length(T_ann) - length(T_ann) // 10)
    T_ann[1:n_ann] = collect(range(4.0, T, length=n_ann))
    T_ann[n_ann:end] .= T
    T_ann = vcat(T_ann, [T for _ in 1:N_steps])
    frames = @animate for i in eachindex(T_ann)
        global lattice = metropolis_step(lattice, T_ann[i], J, true)
        plot_grid(lattice, i, T_ann[i])
        next!(p)
    end
else
    println("\n Starting Simulation without Annealing ...")
    frames = @animate for i in 1:N_steps
        global lattice = metropolis_step(lattice, T, J, true)
        plot_grid(lattice, i, T)
        next!(p)
    end
end
println("\n Done! \n Saving the animation as a GIF...")
gif(frames, "figs/ising_animation.gif", fps=30)