"""Paper algorithm components; no experiment execution or stored observations."""
import torch

def _check(probability, atoms, active, q):
    if probability.shape != atoms.shape or probability.ndim < 2:
        raise ValueError('Expected identical (...,region,atom) shapes')
    if active.shape != probability.shape[:-1] or active.dtype != torch.bool:
        raise ValueError('Explicit supported-region boolean mask required')
    if q.shape != probability.shape[:-2]:
        raise ValueError('One fixed mixing coefficient per distribution')
    if not torch.isfinite(q).all() or ((q < 0) | (q > 1)).any():
        raise ValueError('q must be in [0,1]')
    selected = probability[active]
    selected_atoms = atoms[active]
    if not torch.isfinite(selected).all() or (selected < 0).any() or (not torch.isfinite(selected_atoms).all()):
        raise ValueError('Supported marginals must be finite nonnegative distributions')
    if not torch.allclose(selected.sum(-1), torch.ones_like(selected.sum(-1)), atol=2e-06, rtol=0):
        raise ValueError('Supported marginal probabilities must sum to one')

def maximum_cdf_survival(probability, atoms, active, q, query, validate=True):
    """Return (...,query) CDF/survival and (...,) eligibility.

    Invalid all-missing rows return NaNs, not a fictitious zero-risk forecast.
    Direct survival summation preserves very small positive probabilities.
    """
    if validate:
        _check(probability, atoms, active, q)
    if query.shape[:-1] != probability.shape[:-2] or not torch.isfinite(query).all():
        raise ValueError('Finite (...,query) values required')
    p = torch.where(active[..., None], probability, 0)
    a = torch.where(active[..., None], atoms, 0)
    right = a[..., None] <= query[..., None, None, :]
    cdf = (p[..., None] * right).sum(-2)
    survival = (p[..., None] * ~right).sum(-2)
    cdf = torch.where(active[..., None], cdf, torch.ones_like(cdf)).clamp(0, 1)
    survival = torch.where(active[..., None], survival, torch.zeros_like(survival)).clamp(0, 1)
    synchronous_cdf = cdf.min(-2).values
    synchronous_survival = survival.max(-2).values
    independent_cdf = cdf.prod(-2)
    small = survival.max(-2).values < 0.5
    safe = torch.where(small[..., None, :], survival, torch.zeros_like(survival))
    stable = -torch.expm1(torch.log1p(-safe).sum(-2))
    independent_survival = torch.where(small, stable, 1 - independent_cdf)
    value = q[..., None] * synchronous_cdf + (1 - q[..., None]) * independent_cdf
    tail = q[..., None] * synchronous_survival + (1 - q[..., None]) * independent_survival
    known = active.any(-1)
    value = torch.where(known[..., None], value, torch.full_like(value, float('nan')))
    tail = torch.where(known[..., None], tail, torch.full_like(tail, float('nan')))
    return (value, tail, known)

def maximum_atomic_distribution(probability, atoms, active, q, validate=True):
    """Exact finite maximum support, up to floating point roundoff.

    Returned probabilities for all-missing rows are all zero AND known=False;
    these rows must be masked. They are not normalized valid distributions.
    """
    if validate:
        _check(probability, atoms, active, q)
    support = torch.where(active[..., None], atoms, torch.zeros_like(atoms)).flatten(-2).sort(-1).values
    cdf, tail, known = maximum_cdf_survival(probability, atoms, active, q, support, validate=False)
    cdf = torch.where(known[..., None], cdf, 0)
    tail = torch.where(known[..., None], tail, 0)
    lower = torch.cat((torch.zeros_like(cdf[..., :1]), cdf[..., :-1]), -1)
    upper = torch.cat((torch.ones_like(tail[..., :1]), tail[..., :-1]), -1)
    mass = torch.where(cdf <= 0.5, cdf - lower, upper - tail)
    if (mass[known] < -3e-06).any():
        raise ArithmeticError('Maximum CDF is not monotone')
    mass = mass.clamp_min(0)
    mass = torch.where(known[..., None], mass, 0)
    total = mass.sum(-1, keepdim=True)
    mass = mass / torch.where(known[..., None], total, torch.ones_like(total))
    return dict(probability=mass, atoms=support, known=known, mean=(mass * support).sum(-1), cdf_at_support=cdf, survival_at_support=tail)
