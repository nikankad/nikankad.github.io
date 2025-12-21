---
title: "Flappy bird using NEAT"
description: "AI that learns flappy bird using the NEAT model"
publishedAt: 2025-12-19T12:00:00Z
tags: ["Machine Learning", "NEAT"]
---

<div style="display: flex; gap: 1rem; margin-bottom: 0.2rem;">
  <img src="https://media.cnn.com/api/v1/images/stellar/prod/140204204156-flappy-bird.jpg?q=w_1280,h_720,x_0,y_0,c_fill" alt="Flappy Bird" style="flex: 1; max-width: 300px;" />
  <img src="https://www.oreilly.com/api/v2/epubs/urn:orm:book:9781838824914/files/assets/79b7b32f-0c4e-4871-9773-006c9c4ff857.png" alt="NEAT Network" style="flex: 1; max-width: 300px;" />
</div>

*Image sources: <a href="https://www.cnn.com/2014/02/05/tech/gaming-gadgets/flappy-bird-game" target="_blank">CNN</a>, <a href="https://www.oreilly.com/library/view/hands-on-neuroevolution-with/9781838824914/76ecb37a-c4b4-4d72-b4fa-449597f3b654.xhtml" target="_blank">O'Reilly</a>*

## What is NEAT?

NeuroEvolution of Augmenting Topologies (NEAT) evolves both neural network weights and structure. It solves three key problems: crossover without data loss, protecting new innovations through speciation, and maintaining simple topologies without manual complexity penalties.

Each genome contains connection genes with innovation numbers that enable $O(n)$ gene alignment during crossover. Networks grow via two mutations: adding connections between nodes, or splitting connections to insert new nodes. Species compete within niches defined by a compatibility threshold $\delta_t$, allowing novel structures time to optimize.

---

## Flappy Bird Implementation

### Game & Neural Network
Built with `pygame` (400×600 screen, gravity $g=2.0$, flap strength $v=-20$). The network receives 4 inputs: bird y-position, distance to pipe, distance to gap center, and velocity (all normalized). Output > 0.0 triggers a flap.

### Fitness & Evolution
* +0.001 per frame, +5 per pipe, +10 at score 20, -4 for collision
* Runs 200 generations using NEAT-Python
* Each genome controls one bird; dead birds are removed when they hit pipes, ground, or bounds

NEAT evolves efficient network topologies that master the game through survival-based selection.