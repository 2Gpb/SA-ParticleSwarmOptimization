# SelfAdaptive-ParticleSwarmOptimization (SA-PSO)

This project implements the Self-Adaptive Particle Swarm Optimization (SA-PSO) algorithm, which combines the concepts of Particle Swarm Optimization (PSO) and Self-Adaptive mechanisms. The main feature of SA-PSO is the dynamic adaptation of the inertia weight w during the optimization process, improving the balance between exploration and exploitation.

### Main Components
- **Particle Class** — Represents an individual particle with methods to update its velocity, position, and track the best position based on fitness.
- **SAParticleSwarmOptimization Class** — Manages the swarm, dynamically adjusts the inertia weight, and iteratively updates particles’ positions and velocities to find an optimal solution.

### Getting Started:
To run this project, you need to install the required dependencies. You can install them using pip: 
```
pip install -r requirements.txt
```
---
