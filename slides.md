% title: Statistical Modeling of Conformational Dynamics
% subtitle: Second Year Progress Report
% author: Robert McGibbon
% author: August 27, 2013
% thankyou: Questions?
% thankyou_details: Thanks especially to Vijay, Christian S., TJ L. and Kyle B.
% contact: <span>www</span> <a href="http://rmcgibbo.appspot.com/">website</a>
% contact: <span>github</span> <a href="https://github.com/rmcgibbo">rmcgibbo</a>
% favicon: http://www.stanford.edu/favicon.ico

---
title: Overview
dsubtitle: Three Parts

<div style="float:right; margin-top:-20px">
  <img height=190px src="figures/folding-mosaic.jpg"/>
  <img height=210px src="figures/hingeloss.png"/>
  <br/>
  <img height=200px style="padding-top:10px;" src="figures/msmaccelerator.png"/>
</div>

- Motivation
- Improving MSM construction with large-margin metric learning
- Current projects and future directions
    - Adaptive sampling
    - Statistical model selection
    - Richer model classes
    
---
title: Biology at the Atomic Length Scale

<center><img height=400  src="figures/U1CP1-1_SizeScale_ksm.jpg" /></center>
<footer class="source"> http://www.nature.com/scitable/topicpage/what-is-a-cell-14023083</footer>

---
title: Biology at the Atomic Length Scale

<center>
<table><tr>
  <td> <img height=200 src="figures/Mallacci-Brain-122-1823-1999-image.png"/> </td>
  <td> <img width=600 src="figures/Mallacci-Brain-122-1823-1999-title.png"/> </td>
</tr></table>
<img width=500 src="figures/Alanine-Valine.png"/>
</center>

<footer class="source">
G.R Mallacci et. al., Brain 122, 1823 (1999).
</footer>

---
title: Molecular Dynamics

<center>
<img style="float:left;" height=400 src="figures/vdw-protein-water.png"/>
<div style="margin:5  0px">
  <img height=150 src="figures/amber-functional-form.png"/>
</div>
</center>

- Calculate the physical interactions in the system.
- Numerically integrate the equations of motion.

<footer class="source">
W. D. Cornell et. al., J. Am. Chem. Soc. 117, 5179 (1995).
</footer>

---
title: MD Datasets are Large
subtitle: First world problems

<center>
<img height=200 src="figures/5348951193_b53fa19c23.jpg"/>
<img width=250 style="margin-left:20px; margin-right:20px" src="figures/TitanNew-bg.jpg"/>
<img height=200 src="figures/folding-mosaic.jpg"/>
</center>

- $100 \frac{\text{ns}}{\text{day } \cdot \text{ GPU}} \cdot 500 \text{ GPUs} \cdot 1 \text{ week} = 350 \text{ $\mu$s}$
- Storing the positions every 200 ps, this is a $\sim$ 1 TB dataset.

---
title: Predictive and interpretable models from atomic-level simulations
build_lists: true

- What are the relevant conformational states?
- What are the characteristic dynamics between them?

- States are a Voronoi tessellation of conformation space $$s_i = \lbrace x \in \Omega : d(x, y_i) \lt d(x, y_j) \;\forall\; j \in S, j \neq i \rbrace $$
- Dynamics are Markovian through state space $$P(s_t | s_{t-1}, s_{t-2}, \ldots) = P(s_t | s_{t-1})$$

---
title: Improving Markov State Model Construction
subtitle: "Learning Kinetic Distance Metrics", JCTC (2013)
class: segue dark nobackground


---
title: MSMs have Competing Sources of Error

The MSM state decomposition, a *clustering*, is characterized by a bias-variance trade off.

- **Bias:** Lowering the number of states introduces systematic error in the model's dynamics.
- Hamiltonian mechanics is perfectly Markovian in $\mathbb{R}^{6N}$
- **Variance:** Raising the number of states increases statistical noise in the model's dynamics.
- How do we balance this trade off and avoid overfitting?


---
title: Choosing the States' Shape

<div style="margin-top:50px; float: right;">
<img height=300 src=figures/gpcr_activation.png />
</div>

- Conformational change is characterized by slow *conformationally subtle*
  transitions.
- To resolve these transitions in our models, our states need to be "smaller".
- We can save our statistics by picking their **shape** more intelligently.


---
title: Large-Margin Classification

<img style="float:right" height=250 src=figures/hingeloss.png />


- Goal of the distance metric for clustering is to distinguish *kinetically*-close
  from *kinetically*-far pairs of conformations. 
- Large-margin learning theory: reduce generalization error by separating the
  two classes as far as possible.

$$ \max_{\mathbf{X},\rho} \left[ \alpha \rho - \frac{1}{N} \sum_i^N \lambda_\text{huber} \left(d^\mathbf{X}(\vec{a}_i,\vec{c}_i) - d^\mathbf{X}(\vec{a}_i, \vec{b}_i) - \rho \right) \right] $$

---
title: Optimization and Constraints

$$ d^{\mathbf{X}}(\vec{a}, \vec{b}) = (\vec{a} - \vec{b})^{T} \mathbf{X} (\vec{a} - \vec{b}) $$

$$ \max_{\mathbf{X},\rho} \left[ \alpha \rho - \frac{1}{N} \sum_i^N \lambda \left(d^\mathbf{X}(\vec{a}_i,\vec{c}_i) - d^\mathbf{X}(\vec{a}_i, \vec{b}_i) - \rho \right) \right] $$

- The matrix $\mathbf{X}$ is constrained to be positive semidefinite.
- Relatively efficient optimization by gradient descent with rank-1 updates naturally maintains p.s.d. constraint.

<footer class="source">Shen, C.; Kim, J.; Wang, L. Scalable large-margin Mahalanobis distance metric learning. IEEE Trans. Neural Networks 2010, 21, 1524–1530</footer>

---
title: KDML Model System
class: img-top-center

<img height=350 src="figures/toy_microstates.png" />

2D Brownian dynamics, where vertical diffusion constant is 10x greater than the horizontal diffusion constant.

<footer class="source">McGibbon, R. T.; Pande, V. S.; J. Chem. Theory Comput., 9 2900 (2013) 10.1021/ct400132h </footer>

---
title: KDML Model System
class: img-top-center

<img height=350 src="figures/timescales.png" />

KDML distance metric gives converged behavior with fewer states.

<footer class="source">McGibbon, R. T.; Pande, V. S.; J. Chem. Theory Comput., 9 2900 (2013) 10.1021/ct400132h </footer>

---
title: Fip35 WW Domain

<table><tr>
<td><img height=250 src="figures/bars.png" /></td>
<td><img height=250 src="figures/state12.png" /></td>
</tr></table>

- The folding timescale is remarkably robust to changes in the distance metric.
- New timescales are observed in the 100 ns - 1 μs regime, corresponding to near-native hydrogen bond reorganizations in the turns.

---
title: Future Directions
subtitle: Adaptive sampling, model selection, statistical learning.
class: segue dark nobackground

---
title: Current projects
subtitle: MSM-accelerated Distributed Molecular Dynamics

<div style="margin-top:-120px; float:right;">
<table>
<tr><td> <img height=150 src="figures/muller.png" /> </td></tr>
<tr><td> <img  height=150 src="figures/villin.native.png" /> </td></tr>
<tr><td> <img  height=150 src="figures/HIV1-cropped.png" /> </td></tr>
</table>
</div>

- Node-parallelism is the present and future of computing. We must exploit ergodic theorem.
- MSMAccelerator: cluster based client-server architecture over ZeroMQ. 
- Runs simulations with OpenMM & AMBER.
- Starting conditions determined on-the-fly by MSMBuilder.

<footer class="source">McGibbon, R.T.; Kiss, G.; Harrigan, M. P; Pande, V. S., <i>in preparation</i></footer>

---
title: Current Projects
subtitle: Hierarchical Bayesian Mutant Sampling

<div style="margin-top:-120px; float:right;">
<img width=400 src="figures/cayley.png" /><br/>
<img width=400 src="figures/information_gain.png" /> 
</div>

- Informative prior on the mutant based on simulations of the wild-type
$$\vec{p}_i^{M} \sim \operatorname{Dir}(q_i \cdot \vec{c}_i^{WT} + 1/2) $$
- Where $q_i$ models info. transfer between wild-type and mutant states $i$
  with hyperprior: $q_i \sim \text{Beta}(\alpha, \beta)$
- Per-state expected information gain is semi-analytically solvable.

<footer class="source">McGibbon, R.T.; Pande, V. S., <i>in preparation</i></footer>

---
title: Current Projects
subtitle: Optimal MSM Model Selection

<div style="margin-top:-140px;  margin-right:-50px; float:right;">
<img width=220px src="figures/like_comp.png"/>
</div>

<div style="margin-top:-130px; float:right;">
<img width=350px src="figures/overfitting.png"/>
</div>

- Chapman–Kolmogorov tests cannot be used as an objective function.

<div style="float:right; margin-top:-120px;">
$$ T(n \cdot \tau) = T(\tau)^n $$
</div>

<div style="float:right; font-size:80%;">
</div>

- Likelihood function *opens a door*.
    - BIC, Cross validation
    

<div style="font-size:80%; margin-top:-20px">
$$ P(\text{traj} \;|\; \text{MSM}) = \prod_{i=1}^{N} \overbrace{p(x_i | s_i)}^\text{tricky part} \cdot T_{s_{i-1} \rightarrow s_i}  $$
</div>


<footer class="source">Schwantes, C.R.<sup>*</sup>; McGibbon, R.T.<sup>*</sup>; Pande, V.S., <i>in preparation</i></footer>


---
title: New Idea
subtitle: Markov-switching Autoregressive Model

<div style="margin-top:-150px; float:right;">
<img width=550 src="figures/hamilton1990.png" />
</div>

- MSM description of within-state dynamics as i.i.d. samples pushes lagtime out, lowers temporal resolution.
- Hybrid model: dynamics are an Ornstein–Uhlenbeck process, but $\mu, \Sigma, \gamma$
  evolve by latent discrete-state Markov jump process.

<div style="margin-top:-30px; font-size:80%">
$$ P(s_t = j| s_{t-1} = i) = T_{ij} $$

$$ X_{t} = \boldsymbol{A_{s_t}} (X_{t-1}-\mu_{s_t}) + \mathcal{N}(\mu_{s_t}, \boldsymbol{\Sigma}_{s_t}) $$
</div>


<aside class="note">
a,b,c
</aside>

<footer class="source">
Hamilton, J. D. <i>Econometrica</i> 57 (1989): 357-384. <br/>
Hamilton, J. D. <i>J. Econometrics</i> 45 (1990): 39-70.
</footer>

---
title: New Idea
subtitle: Markov-switching Autoregressive Model

<div style="margin-top:-80px; float:right;">
<video width="500" height="500" controls>
  <source src="videos/MSARMvsHMM.mp4" type="video/mp4">
  <source src="videos/MSARMvsHMM.ogg" type="video/ogg">
Your browser does not support the video tag.
</video>
</div>

Realizations from MSArM and (Gaussian) MSM

- Same transition matrix.
- Same within-state equilibrium distributions.
- Which looks more like conformational dynamics?




