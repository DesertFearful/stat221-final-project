# Learning the Wasserstein-Closest Distribution under Independence Constraints

**Authors:** Sasha Stasovskyi, Stefan Nicov, Darius Mardaru

## Overview

Many statistical models simplify multivariate structure by assuming independence, even when the true distribution is strongly dependent. This project studies the problem of replacing a dependent distribution $P$ on $\mathbb{R}^d$ by the independent distribution that is closest to it in Wasserstein distance.

Formally, if $\mathcal{I}$ denotes the class of distributions with independent coordinates, we study

$$
Q^\star \in \arg\min_{Q \in \mathcal{I}} W(P,Q),
$$

where $W(P,Q)$ is the Wasserstein distance.

The main idea is to learn this constrained approximation directly from data using a WGAN-style objective, while restricting the generator so that the output distribution is independent by construction. Instead of first estimating the full joint law and then imposing independence afterward, we optimize directly over an independent model class.

## Research Question

The goal is to understand whether the Wasserstein-closest independent approximation can be learned effectively from samples, and how it differs from standard independence baselines such as the product of marginals.

We ask,

1. Can a constrained generative model learn an approximation to

$$
\arg\min_{Q \in \mathcal{I}} W(P,Q)
$$

directly from data?

2. Is the product distribution

$$
P_1 \otimes \cdots \otimes P_d
$$

actually optimal under Wasserstein distance, or does the optimal independent approximation generally alter the marginals?

3. In tractable settings such as Gaussian models, can the Wasserstein-closest independent distribution be characterized analytically?

## Method

Our method is based on a Wasserstein GAN objective. Given data $X_1,\dots,X_n \sim P$, we consider

$$
\min_\theta \max_{f \in \mathrm{Lip}(1)}
\left\{
\frac{1}{n}\sum_{i=1}^n f(X_i)
-
\mathbb{E}\bigl[f(G_\theta(Z))\bigr]
\right\}.
$$

The key restriction is that the generator is factorized coordinate-wise:

$$
G_\theta(Z)
=
\bigl(G_{1,\theta_1}(Z_1),\dots,G_{d,\theta_d}(Z_d)\bigr),
$$

where $Z_1,\dots,Z_d$ are independent latent variables.

Because each coordinate is generated from an independent latent source, the model distribution $Q_\theta$ is independent by construction. This turns the WGAN into a constrained optimization procedure for approximating the Wasserstein projection of $P$ onto the class $\mathcal{I}$.