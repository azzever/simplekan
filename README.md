# Kolmogorov-Arnold Networks (KAN) Simplification

In this project, we explore the concept of Kolmogorov-Arnold Networks (KAN) and how they utilize learnable functions to represent and approximate functions. In comparison to the original [KAN: Kolmogorov-Arnold Networks](https://arxiv.org/abs/2404.19756), we simplify the KAN layer so it performs as a nonlinear feature expansion (similar to using $x, x^2, x^3, \ldots$) followed by a linear layer, thus making it scalable.

## Kolmogorov-Arnold Representation Theorem

The works of [Vladimir Arnold](https://en.wikipedia.org/wiki/Vladimir_Arnold) and [Andrey Kolmogorov](https://en.wikipedia.org/wiki/Andrey_Kolmogorov) established that if $f$ is a multivariate continuous function, then $f$ can be written as a finite composition of continuous functions of a single variable and the binary operation of addition. More specifically,

$$f(\mathbf{x}) = f(x_1, \ldots, x_n) = \sum_{q=0}^{2n} \Phi_q \left( \sum_{p=1}^{n} \varphi_{q,p}(x_p) \right),$$

where $\varphi_{q,p}: [0, 1] \rightarrow \mathbb{R}$ and $\Phi_q: \mathbb{R} \rightarrow \mathbb{R}$.

Unlike the universal approximation theorem, which learns linear transformations with linear functions as free parameters, the [Kolmogorov-Arnold representation theorem](https://en.wikipedia.org/wiki/Kolmogorov%E2%80%93Arnold_representation_theorem) allows learning 1D functions.

## Parameterizing the 1D Function

In the context of KAN, as described in the paper [KAN: Kolmogorov-Arnold Networks](https://arxiv.org/abs/2404.19756), 1D B-splines are used to parameterize these functions. Each spline can be considered as a combination of basis functions with associated coefficients. The resulting spline is essentially a sum of these basis functions multiplied by their respective coefficients.

## Modification

### Symmetric Cubic Interpolation as Spline Basis Function

We will use symmetric cubic interpolation as the basis functions, which interpolate from 1 to 0 on a range [0, 1]: $2x^3 - 3x^2 + 1$, reflected by the y-axis. This allows efficient (non-recurrent) evaluation of basis functions on the input.

### Fixed Input Range and Squashing Function

We apply a squashing function (like tanh) to restrict the input to be in the range $[-1, 1]$. This allows us to keep a fixed grid range for all layers and not worry about normalization. Thus each spline (function) is really a composition of squashing function with learned function `Spline(tanh(x), spline_params)`. For symbolic analysis, this function can be fused into the spline with handling grid positions manually.

### Single Matrix Multiplication

The Kolmogorov-Arnold representation theorem involves a summation operation, and each spline is also a sum of basis functions. This allows us to perform the summation as a single linear operation (summing over all basis functions and inputs).
