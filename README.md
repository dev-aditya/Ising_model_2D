# Markov Chain Monte Carlo Simulation of 2D Ising Model

This repository contains a Julia implementation of a Markov Chain Monte Carlo (MCMC) simulation of the 2D Ising model. It was originally created for a PHY644 Term Paper.

## Features

- Simulation of the Ising model on a 2D lattice.
- Analysis of various physical properties like specific heat capacity, average magnetization, average energy, and magnetic susceptibility over a range of temperatures.
- Utilization of multi-threading for performance improvement.
- Progress tracking of the simulation runs.

## Setup
_Anyone cloning this repository can recreate the exact environment by navigating to the project directory and activating the project:_
```shell
cd ising_model
julia --project=.
```
_And then instantiating the environment to download all dependencies:_
```julia
using Pkg
Pkg.instantiate()
```