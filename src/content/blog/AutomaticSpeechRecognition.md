---
title: "IBNet: Inverted Bottleneck Network for Lightweight Automatic Speech Recognition"
description: "A new end-to-end convolutional ASR model that integrates inverted bottleneck modules into the time-channel separable convolution framework, achieving competitive accuracy at a fraction of the cost."
publishedAt: 2026-04-16T00:00:00Z
draft: false
---

## Background

Automatic speech recognition (ASR) is everywhere — it powers virtual assistants, real-time captions, voice navigation, and accessibility tools. The field has made enormous strides in accuracy over the past decade, but most of those gains came from models that are simply too large and compute-hungry to run on a phone or embedded device.

The standard playbook in modern ASR involves Transformer-based architectures like the Conformer, which combine self-attention with convolution to capture both global and local patterns in speech. These models achieve impressive word error rates (WER) on benchmarks like LibriSpeech, but they often exceed 100 million parameters — far too heavy for on-device inference.

Our goal with **IBNet** was to close that gap: build an end-to-end ASR model that is compact enough for edge deployment without sacrificing the recognition accuracy that makes these systems actually useful.

---

## Starting Point: QuartzNet

IBNet builds directly on **QuartzNet**, a lightweight ASR model from NVIDIA that introduced 1D time-channel separable convolutions to the ASR domain. QuartzNet replaces the expensive 2D convolutions used in earlier acoustic models with factored 1D depthwise + pointwise convolutions, dramatically reducing parameter count while maintaining competitive WER.

The QuartzNet architecture consists of:
- An initial strided convolutional layer for temporal downsampling
- Five block groups (B1–B5), each containing R repeated depthwise-separable modules with progressively wider kernels
- Block-level residual connections via 1×1 pointwise projections
- Two output pointwise layers before CTC decoding

Three size variants exist: **5×5** (6.7M params), **10×5** (12.8M), and **15×5** (18.9M). At under 20M parameters, QuartzNet was already a strong baseline for efficient ASR.

The limitation we identified: every module in QuartzNet operates at a **fixed channel dimensionality**. The depthwise temporal convolution sees exactly as many channels as came in — no more, no less. This constrains how diverse the temporal features can be at each layer.

---

## The IBNet Idea

IBNet's core insight is borrowed from MobileNetV2: instead of doing depthwise convolution at the input channel dimensionality, **expand first, convolve in the wider space, then compress back**.

This "expand–convolve–compress" pattern, applied to the 1D temporal domain, gives the depthwise convolution access to a richer intermediate representation without increasing the input or output channel count. The result is a more expressive feature extractor at a comparable parameter budget.

We call the fundamental unit an **Inverted Bottleneck Convolution (IBConv)** module.

---

## Architecture

### Overall Structure

The IBNet architecture mirrors QuartzNet's high-level layout but replaces every time-channel separable module with an IBConv module:

```
Input (n_mels × T)
    ↓
C1: Conv-BN-ReLU  [stride 2, kernel 33]   → C channels
    ↓
B1: IBBlock  [K=33]  C → C
B2: IBBlock  [K=39]  C → C
B3: IBBlock  [K=51]  C → 2C
B4: IBBlock  [K=63]  2C → 2C
B5: IBBlock  [K=75]  2C → 2C
    ↓
C2: IBConv   [K=87]  2C
C3: Conv-BN-ReLU     → 4C
C4: Conv (dilation=2) → |labels|
    ↓
CTC Loss
```

The progressive kernel widening from K=33 to K=87 reflects how speech operates at multiple timescales: narrow kernels capture fine phonetic detail, while wide kernels capture prosodic and co-articulation patterns that span longer stretches of audio.

Blocks B1–B2 operate at channel width C. Blocks B3–B5 double to 2C. This mirrors common practice in deep CNNs: early layers handle low-level acoustic features (spectral envelopes, formant transitions) that don't need many channels, while deeper layers represent higher-level linguistic abstractions (phoneme sequences, word boundaries) that benefit from a richer feature space. Doubling at B3 coincides with the shift to wider kernels (K=51), so the network simultaneously gains both broader temporal context and greater channel capacity.

### IBConv Module

Each IBConv module transforms an input tensor **X** ∈ ℝ^(C_in × T) to an output **Y** ∈ ℝ^(C_out × T) through three stages:

**1. Pointwise Expansion**

A 1×1 convolution expands from C_in to C_mid = C_in × t channels (where t is the expansion ratio), followed by BatchNorm and ReLU:

```
H1 = ReLU(BN(Conv1×1(X)))    ∈ ℝ^(C_mid × T)
```

**2. Depthwise Temporal Convolution**

A K-wide depthwise 1D convolution operates on H1 independently per channel, followed by BatchNorm and ReLU:

```
H2 = ReLU(BN(DWConv_K(H1)))  ∈ ℝ^(C_mid 
```

**3. Pointwise Compression — Linear Bottleneck**

A 1×1 convolution compresses from C_mid back to C_out, followed by BatchNorm but **no ReLU**:

```
Ŷ = BN(Conv1×1(H2))          ∈ ℝ^(C_out × T)
```

When C_in = C_out, a per-module residual is added:

```
Y = Ŷ + X   (if C_in == C_out)
Y = Ŷ       (otherwise)
```

The omission of ReLU after compression is critical. In the narrow output space, ReLU would zero out roughly half of all activations — irreversibly discarding information. In the wider intermediate space, there is sufficient redundancy that ReLU can be applied without significant information loss. This is the "linear bottleneck" insight from MobileNetV2, adapted here to the temporal audio domain.

### IBBlock

Each IBBlock stacks R IBConv modules and adds a **block-level residual** via a pointwise projection:

```
Z = ReLU( F(X) + BN(Conv1×1(X)) )
```

where F(X) = IBConv_R( ··· IBConv_1(X) ) is the sequential output of R stacked modules.

This **dual residual scheme** — per-module shortcuts inside each IBConv plus a block-level shortcut across the entire stack — provides two complementary gradient pathways:

- **Module-level residuals** help each IBConv learn incremental refinements to its input
- **Block-level residuals** ensure gradients can flow directly from the block output back to the block input, stabilizing training in deep stacks

This is particularly important for IBNet because each IBConv already contains three convolution layers (expand, depthwise, compress), making the effective depth within a single block substantially greater than in QuartzNet.

---

## Why Inverted Bottlenecks Work for ASR

Standard depthwise separable convolutions, as used in QuartzNet, perform temporal filtering at the same channel width as the input. The number of distinct temporal patterns the network can simultaneously model is constrained by C.

By expanding to C_mid = t × C before the depthwise step, the convolution operates in a space with t times as many independent filter channels. This allows the network to learn a richer, more diverse set of temporal features — different channels can specialize in different acoustic phenomena — while the subsequent compression keeps the output compact and parameter count reasonable.

The tradeoff is a modest increase in computation during the expand/compress steps, but since pointwise convolutions have no kernel width, they are much cheaper than wide depthwise convolutions.

---

## Training Setup

IBNet is trained end-to-end with **CTC loss** on the LibriSpeech `train-clean-100` subset (~100 hours of clean English speech). This controlled setting isolates architectural improvements from data-scale effects and allows systematic comparison under a fixed compute budget.

We evaluate against QuartzNet baselines under four configurations:
- **Greedy decoding** — no language model, argmax at each frame
- **SpecCutout only** — spectrogram-level data augmentation
- **Speed Perturbation only** — audio speed perturbation augmentation
- **SpecCutout + LM** — augmentation plus a shallow language model at decode time

Evaluation is reported on four splits: `dev-clean`, `dev-other`, `test-clean`, `test-other`.

---

## Results

Experiments are ongoing, but preliminary results show that IBNet (C=172, R=3, t=2) achieves competitive WER against QuartzNet 5×5 (6.7M parameters) under comparable parameter budgets. The dual residual scheme and linear bottleneck design are both critical — ablations removing either component show degraded validation WER.

Full WER numbers across all configurations and model variants will be published here and in the final paper upon experiment completion.

---

## What's Next

A few directions we're actively exploring:

**Broader benchmarks.** The current evaluation is on `train-clean-100`. We plan to extend to the full LibriSpeech 960-hour set and to multilingual and noisy speech datasets to test generalizability.

**On-device deployment.** IBNet's compact size makes it a natural candidate for mobile ASR. We're working on quantizing and exporting the model for real-time on-device inference.

**Augmentation in the expanded space.** One intriguing idea: since IBNet explicitly creates an expanded intermediate feature space, data augmentation could be applied *there* rather than at the input spectrogram level. Whether augmenting in that richer representation improves robustness under domain shift is an open question.

**Integration with language models.** IBNet outputs character-level CTC probabilities. Shallow fusion with n-gram or neural LMs at decode time is a standard technique for improving WER, and we're evaluating several configurations.

---

## Code

The source code and pre-trained models are available at [github.com/nikankad/Notarius](https://github.com/nikankad/Notarius).
