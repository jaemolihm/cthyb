"""
Test for measure_G_tau_impr_est_v2 (improved estimator for G_tau).

Validates:
1. Exact diagonalization comparison (for small system)
2. Self-consistency checks (relations between O1, O2, O1_O2)
3. Regression against reference data

Author: Test for CTHYB v2 improved estimator
"""

import numpy as np
from triqs.gf import *
from triqs.operators import *
from triqs.atom_diag import AtomDiag, atomic_g_iw
from h5 import HDFArchive
from triqs.utility.comparison_tests import assert_block_gfs_are_close
from triqs.utility.h5diff import h5diff
import triqs.utility.mpi as mpi
from triqs_cthyb import Solver
import itertools

# ==============================================================================
# Model Parameters
# ==============================================================================

beta = 5.0
U = 5.0
block_names = ['ud']
gf_struct = [['ud', 2]]
n_tau = 501
n_iw = 30
n_orb = 2
n_orb_bath = 2

# Single-particle Hamiltonian matrix
h0_mat = np.array([[-0.2,  0.1], [ 0.1, -0.3]])

# Bath parameters
V_mat = np.array([[1.0, 0.2], [0.2, 1.0]])  # Hybridization matrix
h0_bath_mat = np.array([[-1.0, 0.5], [0.5, -0.3]])  # Bath energy levels

h_int = U * n('ud', 0) * n('ud', 1)

# ==== Non-Interacting Impurity Green function  ====
mesh = MeshImFreq(beta=beta, S='Fermion', n_max=n_iw)
G0_iw = BlockGf(mesh=mesh, gf_struct=gf_struct)

h_tot_mat = np.block([[h0_mat,         V_mat      ],
                      [V_mat.T.conj(), h0_bath_mat]])
for bl, iw in itertools.product(block_names, mesh):
    G0_iw[bl][iw] = np.linalg.inv(iw.value * np.eye(2*n_orb) - h_tot_mat)[:n_orb, :n_orb]

if False:
    # Run ED benchmark calculation

    # ==== Bath & Coupling hamiltonian ====
    c_dag_vec = { s: np.matrix([[c_dag(s,o) for o in range(n_orb)]]) for s in block_names }
    c_vec =     { s: np.matrix([[c(s,o)] for o in range(n_orb)]) for s in block_names }

    c_dag_bath_vec = { s: np.matrix([[c_dag(s, o) for o in range(n_orb, n_orb + n_orb_bath)]]) for s in block_names }
    c_bath_vec =     { s: np.matrix([[c(s, o)] for o in range(n_orb, n_orb + n_orb_bath)]) for s in block_names }

    h_0 = sum(c_dag_vec[s] * h0_mat * c_vec[s] for s in block_names)[0,0]
    h_bath = sum(c_dag_bath_vec[s] * h0_bath_mat * c_bath_vec[s] for s in block_names)[0,0]
    h_coup = sum(c_dag_vec[s] * V_mat * c_bath_vec[s] + c_dag_bath_vec[s] * V_mat * c_vec[s] for s in block_names)[0,0]

    # ==== Total impurity hamiltonian ====
    h_tot = h_0 + h_int + h_coup + h_bath

    # --------- Construct the AtomDiag Object ----------

    fop_imp  = [(s,o) for s, n in gf_struct for o in range(n)]
    fop_bath = [(s,o) for s, o in itertools.product(block_names, range(n_orb, n_orb + n_orb_bath))]
    fop_tot  = fop_imp + fop_bath

    ad_tot = AtomDiag(h_tot, fop_tot)

    # --------- Calculate the single-particle Green function ----------

    gf_struct_tot = [[s, n_orb + n_orb_bath] for s in block_names]
    G_iw_tot = atomic_g_iw(ad_tot, beta, gf_struct_tot, n_iw)

    name_list = [bl for bl, n_orb in gf_struct]
    block_list = [G_iw_tot[bl][:n_orb, :n_orb] for bl, n_orb in gf_struct]
    G_exact = BlockGf(name_list=name_list, block_list=block_list)

    Sigma_exact = inverse(G0_iw) - inverse(G_exact)

    # Save atom_diag benchmark results
    if mpi.is_master_node():
        with HDFArchive("G_tau_impr_est.bench.h5", 'w') as Results:
            Results["G_iw"] = G_exact
            Results["Sigma_iw"] = Sigma_exact


S = Solver(beta=beta, gf_struct=gf_struct, n_tau=n_tau, n_iw=n_iw)
S.G0_iw << G0_iw

S.solve(
    h_int=h_int,
    measure_G_tau=True,
    measure_G_tau_impr_est_v2=True,
    n_cycles=50_000,
    n_warmup_cycles=1_000,
    random_seed=12345,
    # n_cycles=1_000_000,      # high-precision test
    # n_warmup_cycles=10_000,  # high-precision test
    length_cycle=50,
    measure_density_matrix=True,
    use_norm_as_weight=True,
    measure_pert_order=True,
)


# Check internal self-consistency relations coming from equation of motion
# See Eqs.(27,29) of Phys. Rev. B 109, 125138 (2024)

G_iw_IE_FL_EOM = S.Sigma_iw * S.G_iw
G_iw_IE_FR_EOM = S.G_iw * S.Sigma_iw
G_iw_IE_I_EOM = S.Sigma_iw + G_iw_IE_FL_EOM * inverse(S.G_iw) * G_iw_IE_FR_EOM
for bl, g in G_iw_IE_I_EOM:
    g -= S.Sigma_Hartree[bl]

if mpi.is_master_node():
    w = MatsubaraFreq(0, beta, 'Fermion')

    # The following high-precision tests pass when run with 96 mpi processes, and
    # n_cycles=1_000_000, n_warmup_cycles=10_000, length_cycle=50.
    # np.testing.assert_allclose(S.G_iw_IE_FL['ud'](w), G_iw_IE_FL_EOM['ud'](w), atol=0.0006)
    # np.testing.assert_allclose(S.G_iw_IE_FR['ud'](w), G_iw_IE_FR_EOM['ud'](w), atol=0.0006)
    # np.testing.assert_allclose(S.G_iw_IE_I['ud'](w), G_iw_IE_I_EOM['ud'](w), atol=0.005)

    np.testing.assert_allclose(S.G_iw_IE_FL['ud'](w), G_iw_IE_FL_EOM['ud'](w), atol=0.02)
    np.testing.assert_allclose(S.G_iw_IE_FR['ud'](w), G_iw_IE_FR_EOM['ud'](w), atol=0.02)
    np.testing.assert_allclose(S.G_iw_IE_I['ud'](w), G_iw_IE_I_EOM['ud'](w), atol=0.08)



# Compare with reference data

if mpi.is_master_node():
    filename = 'G_tau_impr_est.out.h5'
    with HDFArchive(filename, 'w') as res:
        res['G_iw_IE_FL'] = S.G_iw_IE_FL
        res['G_iw_IE_FR'] = S.G_iw_IE_FR
        res['G_iw_IE_I'] = S.G_iw_IE_I
        res['Sigma_iw_IE_asym_L'] = S.Sigma_iw_IE_asym_L
        res['Sigma_iw_IE_asym_R'] = S.Sigma_iw_IE_asym_R
        res['Sigma_iw_IE_sym'] = S.Sigma_iw_IE_sym

h5diff(filename, 'G_tau_impr_est.ref.h5')


# Compare with exact diagonalization benchmark

if mpi.is_master_node():
    with HDFArchive("G_tau_impr_est.bench.h5", 'r') as Results:
        Sigma_exact = Results["Sigma_iw"]

    # The following high-precision tests pass when run with 96 mpi processes, and
    # n_cycles=1_000_000, n_warmup_cycles=10_000, length_cycle=50.
    # assert_block_gfs_are_close(S.Sigma_iw, Sigma_exact, 0.4)
    # assert_block_gfs_are_close(S.Sigma_iw_IE_asym_L, Sigma_exact, 0.05)
    # assert_block_gfs_are_close(S.Sigma_iw_IE_asym_R, Sigma_exact, 0.05)
    # assert_block_gfs_are_close(S.Sigma_iw_IE_sym, Sigma_exact, 0.004)

    assert_block_gfs_are_close(S.Sigma_iw_IE_asym_L, Sigma_exact, 0.4)
    assert_block_gfs_are_close(S.Sigma_iw_IE_asym_R, Sigma_exact, 0.4)
    assert_block_gfs_are_close(S.Sigma_iw_IE_sym, Sigma_exact, 0.06)
