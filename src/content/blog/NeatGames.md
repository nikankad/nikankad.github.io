---
title: "Teaching AI to Play Flappy Bird with NEAT"
description: "How I used NeuroEvolution of Augmenting Topologies to train an AI that masters Flappy Bird through genetic algorithms and neural network evolution"
publishedAt: 2025-12-19T12:00:00Z
tags: ["Machine Learning", "NEAT", "Python", "AI", "Game Development"]
---

![Flappy Bird Game](https://media.cnn.com/api/v1/images/stellar/prod/140204204156-flappy-bird.jpg?q=w_1280,h_720,x_0,y_0,c_fill)

*The infamous Flappy Bird game.*

## Introduction

Remember Flappy Bird? The notoriously difficult mobile game that took the world by storm in 2014? I decided to tackle it from a different angle: **what if an AI could learn to play it perfectly?**

This project uses **NEAT (NeuroEvolution of Augmenting Topologies)**, a genetic algorithm that evolves both the structure and weights of neural networks. Unlike traditional neural networks where you define the architecture upfront, NEAT starts simple and grows complexity only when needed.

---

## What is NEAT?

![Genetic Algorithm Flow](https://upload.wikimedia.org/wikipedia/commons/thumb/1/1f/Genetic_algorithm.svg/400px-Genetic_algorithm.svg.png)

*The genetic algorithm cycle: selection, crossover, mutation, and evaluation.*

**NeuroEvolution of Augmenting Topologies** is an evolutionary algorithm developed by Kenneth O. Stanley in 2002. It solves three fundamental challenges in neuroevolution:

### 1. The Competing Conventions Problem

When two neural networks solve the same problem differently, how do you combine them? NEAT uses **innovation numbers** (unique identifiers assigned to each new connection) to enable meaningful crossover between different network topologies.

### 2. Protecting Innovation Through Speciation

New structural mutations often perform poorly at first. NEAT groups similar networks into **species**, allowing innovations time to optimize before competing against the broader population.

### 3. Minimizing Dimensionality

Rather than starting with complex architectures, NEAT begins with minimal networks and adds complexity through mutation only when beneficial. This keeps solutions as simple as possible.

### Key NEAT Mutations

- **Add Connection:** Creates a new link between two previously unconnected nodes
- **Add Node:** Splits an existing connection, inserting a new hidden neuron
- **Weight Mutation:** Adjusts connection weights through perturbation or randomization

---

## The Flappy Bird Challenge

![Flappy Bird Neural Network Inputs](https://miro.medium.com/v2/resize:fit:640/format:webp/1*euTz_X0DyvxePYDIZOeNNQ.png)

*The neural network receives game state information as inputs to decide when to flap.*

Flappy Bird seems simple: tap to flap, avoid pipes. But it's deceptively challenging because:

- **Precise timing** is critical. Flap too early or late and you crash.
- **Constant decision-making.** The bird must evaluate every frame.
- **No recovery.** One mistake ends the game.

This makes it a perfect testbed for AI: clear success criteria, simple inputs, and binary output (flap or don't flap).

---

## My Implementation

### Game Setup

I built the game environment using **Pygame** with these parameters:

```python
# Screen dimensions
SCREEN_WIDTH = 400
SCREEN_HEIGHT = 600

# Physics
GRAVITY = 2.0
FLAP_SPEED = 20
GAME_SPEED = 30

# Obstacles
PIPE_WIDTH = 70
PIPE_HEIGHT = 500
PIPE_GAP = 200

# Performance
FPS = 30
```

### Neural Network Architecture

Each bird is controlled by a neural network that receives **4 normalized inputs**:

| Input | Description |
|-------|-------------|
| `bird_y / SCREEN_HEIGHT` | Bird's vertical position (0 = top, 1 = bottom) |
| `dist_to_pipe / SCREEN_WIDTH` | Horizontal distance to the next pipe |
| `(gap_center - bird_y) / SCREEN_HEIGHT` | Vertical distance to the center of the pipe gap |
| `velocity / 20` | Bird's current vertical velocity |

The network uses **tanh** activation and outputs a single value. If **output > 0.0**, the bird flaps.

> **Network Evolution:** Networks start with 4 inputs directly connected to 1 output (no hidden nodes). The initial connection is partial at 50%, meaning not all input-output pairs start connected. NEAT then evolves the topology, adding nodes and connections as needed.

### Fitness Function

The fitness function guides evolution by rewarding good behavior:

```python
# Survival reward (per frame alive)
genome.fitness += 0.001

# Passing a pipe successfully
genome.fitness += 5.0

# Bonus for reaching 20 pipes
if score >= 20:
    genome.fitness += 10.0

# Penalty for collision (ground, pipes, or out of bounds)
genome.fitness -= 4.0
```

This encourages birds to:

1. **Stay alive** as long as possible
2. **Progress** through pipes
3. **Excel** by reaching high scores

---

## Training Process

![NEAT Training Generations](https://miro.medium.com/v2/resize:fit:720/format:webp/1*BYDJpa6M2rzWNSurvspf8Q.png)

*Multiple birds competing simultaneously. Only the fittest survive to reproduce.*

### Evolution Parameters

Using **NEAT-Python**, I configured evolution with:

| Parameter | Value |
|-----------|-------|
| Population Size | 1000 birds per generation |
| Generations | 200 maximum |
| Fitness Threshold | 500 (terminates early if reached) |
| Compatibility Threshold | 2.2 for species grouping |
| Survival Rate | Top 30% reproduce |
| Elitism | 2 best organisms preserved |
| Max Stagnation | 12 generations before species removal |

### Mutation Rates

| Mutation Type | Rate |
|---------------|------|
| Add Connection | 20% |
| Delete Connection | 5% |
| Add Node | 3% |
| Delete Node | 1% |
| Weight Mutation | 60% |
| Bias Mutation | 30% |

### What Happens Each Generation

1. **Spawn** 1000 birds, each with a unique neural network
2. **Run** the game until all birds die
3. **Evaluate** fitness scores
4. **Select** the best performers
5. **Reproduce** through crossover and mutation
6. **Repeat** with the new population

---

## Results

### Training Milestones

| Generations | Achievement |
|-------------|-------------|
| ~5 | First bird passes a pipe |
| ~15 | Consistent score of 10+ |
| ~30 | Networks play indefinitely |

### Observations

**Early Generations (1-10):** Birds mostly crash immediately. Some random mutations occasionally produce birds that survive a few seconds.

**Middle Generations (10-30):** Networks begin to understand the correlation between pipe distance and flapping. Survival times increase dramatically.

**Late Generations (30+):** Champion networks emerge that can play indefinitely. The evolved topology typically has 4-6 connections and 1-2 hidden nodes.

---

## Key Takeaways

**NEAT is Efficient.** Starting with minimal networks and growing complexity only when needed produces elegant solutions.

**Fitness Design Matters.** The reward structure directly shapes what behaviors evolve. Small tweaks can dramatically change outcomes.

**Speciation Preserves Innovation.** Without speciation, novel mutations would be eliminated before they could prove their worth.

**Large Populations Help.** With 1000 birds per generation, there's enough diversity for evolution to explore many strategies simultaneously.

---

## Try It Yourself

The full source code is available on GitHub:

**[View on GitHub →](https://github.com/nikankad/NEAT-games)**

### Running the Project

```bash
# Clone the repository
git clone https://github.com/nikankad/NEAT-games.git
cd NEAT-games/FlappyBirdAi

# Install dependencies
pip install -r requirements.txt

# Run Flappy Bird NEAT
python game/flappy.py
```

Note: For visualization features (network topology graphs, fitness plots), you'll need Graphviz installed on your system.

---

*Built with Python, Pygame, and NEAT-Python.*
