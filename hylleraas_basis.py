"""
Multi-exponent Hylleraas-type basis set
========================================

Reproduces the basis construction used in:

  A. Sadhukhan, S. Dutta, J.K. Saha,
  "Critical stability and structural properties of screened two-electron
  system in Feshbach resonance state", Eur. Phys. J. D (2019) 73:250.

Trial wavefunction (Eq. 3 of the paper) for the 1S^e state of a
two-electron (Zee) system:

    Psi = sum_i r1^li r2^mi r12^ni
              [ sum_{k1}        C_{i,k1,k1} eta_{k1}(1) eta_{k1}(2)
              + sum_{k1<k2}     C_{i,k1,k2} eta_{k1}(1) eta_{k2}(2) ]
          + exchange

with Slater-type orbitals

    eta_i(j) = exp(-rho_i * r_j)

whose non-linear (screening) parameters rho_i are generated from a
geometric progression (GP), Eq. after (3):

    rho_i = rho_{i-1} * gamma ,   i = 1 .. q

"exchange" means applying the particle-permutation operator P12
(swap electron labels 1 <-> 2, i.e. r1 <-> r2 everywhere) to every
term, which symmetrizes the basis under electron exchange.

For p position-power sets (li, mi, ni) and q exponents, the total
number of (symmetry-independent) basis terms is

    M = [q(q+1)/2] * p

matching M = 225 for q = 9, p = 5 quoted in the article.
"""

import numpy as np
from numpy.polynomial.legendre import leggauss
from numpy.polynomial.laguerre import laggauss
from scipy.linalg import eigh


# ----------------------------------------------------------------------
# 1. Basis-set definition
# ----------------------------------------------------------------------

class HylleraasBasis:
    """
    Multi-exponent Hylleraas-type basis for the 1S^e state of a
    two-electron (Zee) system, as in Eq. (3) of Sadhukhan et al. (2019).

    Parameters
    ----------
    power_sets : list of (li, mi, ni) integer tuples, li,mi,ni >= 0
        The p different sets of powers of r1, r2, r12 (Eq. 3).
    rho0 : float
        Starting non-linear parameter of the geometric progression.
    gamma : float
        Common ratio of the geometric progression (the "stabilization"
        / scaling parameter varied in the stabilization method).
    q : int
        Number of exponents (non-linear parameters) per power set.
    """

    def __init__(self, power_sets, rho0, gamma, q):
        self.power_sets = list(power_sets)
        self.p = len(self.power_sets)
        self.q = q
        self.rho0 = rho0
        self.gamma = gamma

        # Geometric progression of non-linear (screening) parameters:
        # rho_i = rho_0 * gamma^(i-1),  i = 1 .. q
        self.rho = rho0 * gamma ** np.arange(q)

        # Build the list of basis terms: one entry per
        # (power-set index i, k1, k2) with k1 <= k2.
        self.terms = []
        for i, (li, mi, ni) in enumerate(self.power_sets):
            for k1 in range(self.q):
                for k2 in range(k1, self.q):
                    self.terms.append((i, li, mi, ni, k1, k2))

        self.M = len(self.terms)  # = p * q(q+1)/2

    # -- Slater-type orbital -------------------------------------------------
    @staticmethod
    def eta(rho, r):
        """eta_i(j) = exp(-rho_i * r_j)."""
        return np.exp(-rho * r)

    # -- One symmetrized basis function --------------------------------------
    def basis_function(self, term_index, r1, r2, r12):
        """
        Evaluate the (electron-exchange symmetrized) basis function
        Phi_{i,k1,k2}(r1, r2, r12), the building block of Eq. (3):

          k1 <  k2 :
            Phi = r1^li r2^mi r12^ni  eta_k1(r1) eta_k2(r2)
                + r1^mi r2^li r12^ni  eta_k2(r1) eta_k1(r2)

          k1 == k2 :
            Phi = r12^ni exp(-rho_k1 (r1+r2)) * (r1^li r2^mi + r1^mi r2^li)

        r1, r2, r12 may be numpy arrays (broadcastable).
        """
        i, li, mi, ni, k1, k2 = self.terms[term_index]
        rho1, rho2 = self.rho[k1], self.rho[k2]

        direct = (r1 ** li) * (r2 ** mi) * (r12 ** ni) \
            * self.eta(rho1, r1) * self.eta(rho2, r2)
        exchange = (r1 ** mi) * (r2 ** li) * (r12 ** ni) \
            * self.eta(rho2, r1) * self.eta(rho1, r2)

        return direct + exchange

    def evaluate_all(self, r1, r2, r12):
        """Return array of shape (M, ...) with every basis function
        evaluated at the given (r1, r2, r12)."""
        return np.array([self.basis_function(t, r1, r2, r12)
                          for t in range(self.M)])

    def __repr__(self):
        return (f"HylleraasBasis(p={self.p}, q={self.q}, "
                f"M={self.M} terms, rho0={self.rho0}, gamma={self.gamma})")


# ----------------------------------------------------------------------
# 2. Screened Coulomb effective potential (Eq. 2 of the paper)
# ----------------------------------------------------------------------

def V_eff(Z, lam, r1, r2, r12):
    """
    V_eff = -Z [ e^{-lam r1}/r1 + e^{-lam r2}/r2 ] + e^{-lam r12}/r12

    (Yukawa/Debye-screened nucleus-electron attraction and
    electron-electron repulsion, Eq. (2)).
    """
    return (-Z * (np.exp(-lam * r1) / r1 + np.exp(-lam * r2) / r2)
            + np.exp(-lam * r12) / r12)


# ----------------------------------------------------------------------
# 3. Matrix elements via numerical quadrature in perimetric coordinates
# ----------------------------------------------------------------------
#
# Volume element:  dtau = r1 r2 r12 dr1 dr2 dr12 ,
# with the triangle constraint  |r1 - r2| <= r12 <= r1 + r2.
#
# We integrate r1, r2 over (0, inf) with Gauss-Laguerre quadrature and,
# for each (r1, r2) node, integrate r12 over the allowed triangle
# range with Gauss-Legendre quadrature. This gives the overlap matrix
# S and, together with V_eff, the potential-energy part of H exactly
# in the spirit of the paper's variational setup; the kinetic part
# would additionally need the derivative (Fock-type) terms of Eq. (1)
# and is left as an extension point (see `kinetic_matrix_fd` below for
# a simple finite-difference alternative).

def _quadrature_nodes(n_r=24, n_ang=16):
    """Gauss-Laguerre nodes/weights for r1, r2 in (0, inf) and
    Gauss-Legendre nodes/weights (on [-1, 1], rescaled) for r12."""
    x_r, w_r = laggauss(n_r)          # weight exp(-x); r = x
    x_a, w_a = leggauss(n_ang)        # on [-1, 1]
    return x_r, w_r, x_a, w_a


def overlap_matrix(basis, n_r=24, n_ang=16):
    """
    Numerically build the overlap matrix S_{alpha,beta} =
    <Phi_alpha | Phi_beta> using r1, r2 in (0,inf) (Gauss-Laguerre)
    and r12 in [|r1-r2|, r1+r2] (Gauss-Legendre), with the correct
    volume element r1 r2 r12.
    """
    x_r, w_r, x_a, w_a = _quadrature_nodes(n_r, n_ang)
    M = basis.M
    S = np.zeros((M, M))

    for a, wa in zip(x_r, w_r):          # r1 node (Laguerre: true r1 = a)
        r1 = a
        for b, wb in zip(x_r, w_r):      # r2 node
            r2 = b
            lo, hi = abs(r1 - r2), r1 + r2
            if hi <= 0:
                continue
            # map Legendre nodes x_a in [-1,1] to r12 in [lo, hi]
            r12 = 0.5 * (hi - lo) * x_a + 0.5 * (hi + lo)
            jac = 0.5 * (hi - lo)
            vol = r1 * r2 * r12 * jac    # dtau weight (r12 part)

            phi = basis.evaluate_all(r1, r2, r12)   # shape (M, n_ang)
            weight = wa * wb * w_a * vol            # shape (n_ang,)
            S += (phi * weight) @ phi.T

    return S


def potential_matrix(basis, Z, lam, n_r=24, n_ang=16):
    """Potential-energy matrix element  <Phi_alpha | V_eff | Phi_beta>."""
    x_r, w_r, x_a, w_a = _quadrature_nodes(n_r, n_ang)
    M = basis.M
    Vm = np.zeros((M, M))

    for a, wa in zip(x_r, w_r):
        r1 = a
        for b, wb in zip(x_r, w_r):
            r2 = b
            lo, hi = abs(r1 - r2), r1 + r2
            if hi <= 0:
                continue
            r12 = 0.5 * (hi - lo) * x_a + 0.5 * (hi + lo)
            jac = 0.5 * (hi - lo)
            vol = r1 * r2 * r12 * jac

            phi = basis.evaluate_all(r1, r2, r12)
            v = V_eff(Z, lam, r1, r2, r12)
            weight = wa * wb * w_a * vol * v
            Vm += (phi * weight) @ phi.T

    return Vm


# ----------------------------------------------------------------------
# 4. Example usage
# ----------------------------------------------------------------------

if __name__ == "__main__":
    # p = 5 example power sets (li, mi, ni), q = 9 exponents  ->  M = 225,
    # exactly as in the article (q(q+1)/2 * p = 45 * 5 = 225).
    power_sets = [
        (0, 0, 0),
        (1, 0, 0),
        (0, 0, 1),
        (1, 1, 0),
        (0, 0, 2),
    ]

    basis = HylleraasBasis(power_sets, rho0=1.0, gamma=0.7, q=9)
    print(basis)
    print("Total number of basis terms M =", basis.M)

    # sanity check: evaluate the first few basis functions at a point
    r1, r2, r12 = 1.2, 0.8, 1.0
    vals = basis.evaluate_all(np.array(r1), np.array(r2), np.array(r12))
    print("First 5 basis-function values at (r1,r2,r12)="
          f"({r1},{r2},{r12}):")
    print(vals[:5])

    # Build (small, low-accuracy) overlap and potential matrices for a
    # reduced basis to keep the demo fast.
    demo_basis = HylleraasBasis(power_sets[:2], rho0=1.0, gamma=0.7, q=3)
    S = overlap_matrix(demo_basis, n_r=16, n_ang=10)
    Vm = potential_matrix(demo_basis, Z=1.0, lam=0.0, n_r=16, n_ang=10)
    print("\nDemo overlap matrix S shape:", S.shape)
    print("S is symmetric:", np.allclose(S, S.T, atol=1e-8))

    # Generalized eigenproblem  H C = E S C  (Eq. 4).  Here we only show
    # the potential part as a placeholder for H (kinetic terms of Eq. 1
    # must be added for physically meaningful energies).
    evals = eigh(Vm, S, eigvals_only=True)
    print("\nEigenvalues of the (potential-only) demo problem:")
    print(evals)
