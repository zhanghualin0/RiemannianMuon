"""
RiemannianMuon - Muon optimizer adapted for Riemannian manifolds

This optimizer combines:
1. Muon's Newton-Schulz orthogonalization for well-conditioned updates
2. Riemannian geometry to respect manifold constraints

Key differences from standard Muon:
- Converts Euclidean gradients to Riemannian gradients
- Projects orthogonalized updates back to tangent space
- Uses retraction to stay on manifold
- Parallel transports momentum to new tangent space
"""

import torch
import torch.optim

from .mixin import OptimMixin
from geoopt import ManifoldParameter, ManifoldTensor


__all__ = ["RiemannianMuon", "RiemannianMuonWithAuxAdam"]


def zeropower_via_newtonschulz5(G, steps: int = 5):
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G.
    
    This is adapted from Muon's implementation. We use a quintic iteration whose 
    coefficients are selected to maximize the slope at zero.
    
    Parameters
    ----------
    G : torch.Tensor
        Input tensor of shape (..., m, n) to orthogonalize
    steps : int
        Number of Newton-Schulz iterations (default: 5)
    
    Returns
    -------
    torch.Tensor
        Orthogonalized tensor of same shape as G
    """
    assert G.ndim >= 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16() if G.dtype == torch.float32 else G
    
    # Handle tall matrices by transposing
    if G.size(-2) > G.size(-1):
        X = X.mT
    
    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    
    # Perform the Newton-Schulz iterations
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X
    
    if G.size(-2) > G.size(-1):
        X = X.mT
    
    # Convert back to original dtype if needed
    if G.dtype != X.dtype:
        X = X.to(G.dtype)
    
    return X


def riemannian_muon_update(grad, momentum, manifold, point, beta=0.95, ns_steps=5, nesterov=True):
    """
    Compute Riemannian Muon update with tangent space operations.
    
    Parameters
    ----------
    grad : torch.Tensor
        Riemannian gradient in tangent space at point
    momentum : torch.Tensor
        Momentum buffer in tangent space at point
    manifold : geoopt.manifolds.Manifold
        The manifold on which optimization occurs
    point : torch.Tensor
        Current point on the manifold
    beta : float
        Momentum coefficient (default: 0.95)
    ns_steps : int
        Number of Newton-Schulz iterations (default: 5)
    nesterov : bool
        Whether to use Nesterov momentum (default: True)
    
    Returns
    -------
    torch.Tensor
        Update direction in tangent space at point
    """
    # Update momentum in tangent space
    momentum.lerp_(grad, 1 - beta)
    
    # Nesterov lookahead if requested
    update = grad.lerp_(momentum, beta) if nesterov else momentum.clone()
    
    # For 2D or higher dimensional parameters, apply Newton-Schulz orthogonalization
    if update.ndim >= 2:
        original_shape = update.shape
        
        # Handle convolution filters (4D) by reshaping to 2D
        if update.ndim == 4:
            update = update.view(update.size(0), -1)
        elif update.ndim > 2:
            # For other high-dim tensors, flatten all but first dimension
            update = update.view(update.size(0), -1)
        
        # Apply Newton-Schulz orthogonalization
        update_ortho = zeropower_via_newtonschulz5(update, steps=ns_steps)
        
        # Scale adjustment from Kimi.ai (Equation 4 in arXiv:2502.16982)
        # Wt = Wt−1 − ηt (0.2 · Ot · √max(A, B) + λWt−1)
        # This maintains consistent update RMS across different matrix shapes
        A, B = update.size(-2), update.size(-1)
        scale_factor = 0.2 * (max(A, B) ** 0.5)
        update_ortho = update_ortho * scale_factor
        
        # Reshape back to original shape
        update_ortho = update_ortho.reshape(original_shape)
        
        # Project the orthogonalized update back to tangent space
        # This is crucial: Newton-Schulz may take us out of the tangent space
        update = manifold.proju(point, update_ortho)
    
    return update


class RiemannianMuon(OptimMixin, torch.optim.Optimizer):
    """
    Riemannian Muon - MomentUm Orthogonalized by Newton-schulz for Riemannian manifolds.
    
    This optimizer extends Muon to work on Riemannian manifolds by:
    1. Converting Euclidean gradients to Riemannian gradients
    2. Performing momentum updates in tangent spaces
    3. Applying Newton-Schulz orthogonalization to updates
    4. Projecting orthogonalized updates back to tangent space
    5. Using retraction to move along the manifold
    6. Parallel transporting momentum to new tangent spaces
    
    Parameters
    ----------
    params : iterable
        Iterable of parameters to optimize or dicts defining parameter groups
    lr : float
        Learning rate in units of spectral norm per update (default: 0.02)
    momentum : float
        Momentum coefficient (default: 0.95)
    nesterov : bool
        Whether to use Nesterov momentum (default: True)
    ns_steps : int
        Number of Newton-Schulz iterations (default: 5)
    weight_decay : float
        Weight decay coefficient (default: 0)
    stabilize : int
        Stabilize parameters every N steps if not None (default: None)
    
    Notes
    -----
    - For ManifoldParameter/ManifoldTensor, uses the parameter's manifold
    - For regular tensors, treats them as Euclidean
    - Best used for hidden weight matrices (ndim >= 2)
    - Consider using RiemannianAdam for embeddings, biases, and output layers
    
    Example
    -------
    >>> from geoopt import ManifoldParameter
    >>> from geoopt.manifolds import Sphere
    >>> 
    >>> # Create manifold parameter
    >>> manifold = Sphere()
    >>> param = ManifoldParameter(torch.randn(100, 50), manifold=manifold)
    >>> 
    >>> # Create optimizer
    >>> optimizer = RiemannianMuon([param], lr=0.02, momentum=0.95)
    >>> 
    >>> # Training loop
    >>> optimizer.zero_grad()
    >>> loss = compute_loss(param)
    >>> loss.backward()
    >>> optimizer.step()
    """
    
    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, 
                 ns_steps=5, weight_decay=0, stabilize=None):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0 or momentum >= 1.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        
        defaults = dict(
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            weight_decay=weight_decay,
            stabilize=stabilize,
        )
        super().__init__(params, stabilize=stabilize, **defaults)
    
    @torch.no_grad()
    def step(self, closure=None):
        """
        Perform a single optimization step.
        
        Parameters
        ----------
        closure : callable, optional
            A closure that reevaluates the model and returns the loss
        
        Returns
        -------
        loss : torch.Tensor or None
            The loss value if closure is provided
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        for group in self.param_groups:
            momentum_coef = group["momentum"]
            weight_decay = group["weight_decay"]
            learning_rate = group["lr"]
            nesterov = group["nesterov"]
            ns_steps = group["ns_steps"]
            
            stabilize_flag = False
            
            for point in group["params"]:
                grad = point.grad
                if grad is None:
                    continue
                
                # Determine the manifold for this parameter
                if isinstance(point, (ManifoldParameter, ManifoldTensor)):
                    manifold = point.manifold
                else:
                    manifold = self._default_manifold
                
                # Convert Euclidean gradient to Riemannian gradient
                grad = manifold.egrad2rgrad(point, grad)
                
                # Add weight decay to gradient (in tangent space)
                if weight_decay != 0:
                    grad = grad.add(point, alpha=weight_decay)
                
                # Get or initialize state
                state = self.state[point]
                if len(state) == 0:
                    state["step"] = 0
                    # Initialize momentum buffer in tangent space
                    state["momentum_buffer"] = torch.zeros_like(point)
                
                state["step"] += 1
                momentum_buffer = state["momentum_buffer"]
                
                # Compute Riemannian Muon update
                update = riemannian_muon_update(
                    grad=grad,
                    momentum=momentum_buffer,
                    manifold=manifold,
                    point=point,
                    beta=momentum_coef,
                    ns_steps=ns_steps,
                    nesterov=nesterov,
                )
                
                # Retraction + parallel transport
                # Move on manifold and transport momentum to new tangent space
                new_point, momentum_transported = manifold.retr_transp(
                    point, -learning_rate * update, momentum_buffer
                )
                
                # Update parameter and momentum
                point.copy_(new_point)
                momentum_buffer.copy_(momentum_transported)
                
                # Check if stabilization is needed
                if (group["stabilize"] is not None and 
                    state["step"] % group["stabilize"] == 0):
                    stabilize_flag = True
            
            # Stabilize the group if needed
            if stabilize_flag:
                self.stabilize_group(group)
        
        return loss
    
    @torch.no_grad()
    def stabilize_group(self, group):
        """
        Stabilize parameters by projecting them back onto manifolds.
        
        This corrects numerical drift that may accumulate over many steps.
        
        Parameters
        ----------
        group : dict
            Parameter group to stabilize
        """
        for p in group["params"]:
            if not isinstance(p, (ManifoldParameter, ManifoldTensor)):
                continue
            
            state = self.state[p]
            if not state:  # Skip if no state (e.g., due to None grads)
                continue
            
            manifold = p.manifold
            
            # Project point back onto manifold
            p.copy_(manifold.projx(p))
            
            # Project momentum back onto tangent space
            if "momentum_buffer" in state:
                momentum = state["momentum_buffer"]
                momentum.copy_(manifold.proju(p, momentum))


class RiemannianMuonWithAuxAdam(torch.optim.Optimizer):
    """
    Mixed optimizer using RiemannianMuon for manifold parameters and RiemannianAdam for others.
    
    This allows you to use RiemannianMuon for hidden weight matrices on manifolds (ndim >= 2)
    while using RiemannianAdam for embeddings, biases, and other parameters (ndim < 2).
    
    The user must manually specify which parameters shall be optimized with Muon and which 
    with Adam by passing in a list of param_groups with the `use_muon` flag set.
    
    Parameters
    ----------
    param_groups : list of dict
        List of parameter groups, each with a 'use_muon' flag
    
    Example
    -------
    >>> from geoopt import ManifoldParameter
    >>> from geoopt.manifolds import Sphere
    >>> 
    >>> # Separate parameters by type
    >>> hidden_weights = [p for p in model.parameters() 
    ...                  if isinstance(p, ManifoldParameter) and p.ndim >= 2]
    >>> other_params = [p for p in model.parameters() 
    ...                if not (isinstance(p, ManifoldParameter) and p.ndim >= 2)]
    >>> 
    >>> # Setup mixed optimizer
    >>> param_groups = [
    ...     dict(params=hidden_weights, use_muon=True, lr=0.02, momentum=0.95),
    ...     dict(params=other_params, use_muon=False, lr=3e-4, betas=(0.9, 0.999)),
    ... ]
    >>> optimizer = RiemannianMuonWithAuxAdam(param_groups)
    """
    
    def __init__(self, param_groups):
        # Validate and set defaults for each group
        for group in param_groups:
            assert "use_muon" in group, "Each param group must have 'use_muon' flag"
            
            if group["use_muon"]:
                # Muon defaults
                group.setdefault("lr", 0.02)
                group.setdefault("momentum", 0.95)
                group.setdefault("nesterov", True)
                group.setdefault("ns_steps", 5)
                group.setdefault("weight_decay", 0)
                group.setdefault("stabilize", None)
            else:
                # RiemannianAdam defaults
                group.setdefault("lr", 3e-4)
                group.setdefault("betas", (0.9, 0.999))
                group.setdefault("eps", 1e-8)
                group.setdefault("weight_decay", 0)
                group.setdefault("stabilize", None)
        
        super().__init__(param_groups, dict())
        
        # Import here to avoid circular dependency
        from geoopt.manifolds import Euclidean
        self._default_manifold = Euclidean()
    
    @torch.no_grad()
    def step(self, closure=None):
        """Perform a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        for group in self.param_groups:
            if group["use_muon"]:
                self._step_muon(group)
            else:
                self._step_radam(group)
        
        return loss
    
    def _step_muon(self, group):
        """Perform Riemannian Muon step for a parameter group."""
        momentum_coef = group["momentum"]
        weight_decay = group["weight_decay"]
        learning_rate = group["lr"]
        nesterov = group["nesterov"]
        ns_steps = group["ns_steps"]
        
        stabilize_flag = False
        
        for point in group["params"]:
            grad = point.grad
            if grad is None:
                continue
            
            # Determine manifold
            if isinstance(point, (ManifoldParameter, ManifoldTensor)):
                manifold = point.manifold
            else:
                manifold = self._default_manifold
            
            # Convert to Riemannian gradient
            grad = manifold.egrad2rgrad(point, grad)
            
            # Weight decay
            if weight_decay != 0:
                grad = grad.add(point, alpha=weight_decay)
            
            # Initialize state
            state = self.state[point]
            if len(state) == 0:
                state["step"] = 0
                state["momentum_buffer"] = torch.zeros_like(point)
            
            state["step"] += 1
            
            # Compute update
            update = riemannian_muon_update(
                grad=grad,
                momentum=state["momentum_buffer"],
                manifold=manifold,
                point=point,
                beta=momentum_coef,
                ns_steps=ns_steps,
                nesterov=nesterov,
            )
            
            # Retraction + transport
            new_point, momentum_transported = manifold.retr_transp(
                point, -learning_rate * update, state["momentum_buffer"]
            )
            
            point.copy_(new_point)
            state["momentum_buffer"].copy_(momentum_transported)
            
            if (group["stabilize"] is not None and 
                state["step"] % group["stabilize"] == 0):
                stabilize_flag = True
        
        if stabilize_flag:
            self._stabilize_muon_group(group)
    
    def _step_radam(self, group):
        """Perform RiemannianAdam step for a parameter group."""
        lr = group["lr"]
        betas = group["betas"]
        eps = group["eps"]
        weight_decay = group["weight_decay"]
        
        stabilize_flag = False
        
        for point in group["params"]:
            grad = point.grad
            if grad is None:
                continue
            
            # Determine manifold
            if isinstance(point, (ManifoldParameter, ManifoldTensor)):
                manifold = point.manifold
            else:
                manifold = self._default_manifold
            
            state = self.state[point]
            
            # Initialize state
            if len(state) == 0:
                state["step"] = 0
                state["exp_avg"] = torch.zeros_like(point)
                state["exp_avg_sq"] = torch.zeros_like(point)
            
            state["step"] += 1
            exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
            
            # Add weight decay
            if isinstance(point, (ManifoldParameter, ManifoldTensor)) and weight_decay != 0:
                grad = grad.add(point, alpha=weight_decay)
            
            # Convert to Riemannian gradient
            grad = manifold.egrad2rgrad(point, grad)
            
            # Update first and second moments
            exp_avg.mul_(betas[0]).add_(grad, alpha=1 - betas[0])
            exp_avg_sq.mul_(betas[1]).add_(
                manifold.component_inner(point, grad), alpha=1 - betas[1]
            )
            
            # Bias correction
            bias_correction1 = 1 - betas[0] ** state["step"]
            bias_correction2 = 1 - betas[1] ** state["step"]
            
            # Compute direction
            denom = exp_avg_sq.div(bias_correction2).sqrt_()
            direction = exp_avg.div(bias_correction1) / denom.add_(eps)
            
            # Retraction + transport
            new_point, exp_avg_new = manifold.retr_transp(
                point, -lr * direction, exp_avg
            )
            
            point.copy_(new_point)
            exp_avg.copy_(exp_avg_new)
            
            if (group["stabilize"] is not None and 
                state["step"] % group["stabilize"] == 0):
                stabilize_flag = True
        
        if stabilize_flag:
            self._stabilize_radam_group(group)
    
    @torch.no_grad()
    def _stabilize_muon_group(self, group):
        """Stabilize Muon parameters in a group."""
        for p in group["params"]:
            if not isinstance(p, (ManifoldParameter, ManifoldTensor)):
                continue
            
            state = self.state[p]
            if not state:
                continue
            
            manifold = p.manifold
            p.copy_(manifold.projx(p))
            
            if "momentum_buffer" in state:
                state["momentum_buffer"].copy_(
                    manifold.proju(p, state["momentum_buffer"])
                )
    
    @torch.no_grad()
    def _stabilize_radam_group(self, group):
        """Stabilize RiemannianAdam parameters in a group."""
        for p in group["params"]:
            if not isinstance(p, (ManifoldParameter, ManifoldTensor)):
                continue
            
            state = self.state[p]
            if not state:
                continue
            
            manifold = p.manifold
            p.copy_(manifold.projx(p))
            
            if "exp_avg" in state:
                state["exp_avg"].copy_(manifold.proju(p, state["exp_avg"]))

