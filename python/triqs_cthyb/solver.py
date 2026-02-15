################################################################################
#
# TRIQS: a Toolbox for Research in Interacting Quantum Systems
#
# Copyright (C) 2017 by H. UR Strand, P. Seth, I. Krivenko,
#                       M. Ferrero, O. Parcollet
#
# TRIQS is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# TRIQS is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
# details.
#
# You should have received a copy of the GNU General Public License along with
# TRIQS. If not, see <http://www.gnu.org/licenses/>.
#
################################################################################
r"""
the triqs_cthyb solver class
"""
from .solver_core import SolverCore
from triqs.gf import *
import triqs.utility.mpi as mpi
import numpy as np
from itertools import product
from triqs.operators.util.extractors import extract_h_dict, block_matrix_from_op
from .tail_fit import tail_fit as cthyb_tail_fit
from .tail_fit import sigma_high_frequency_moments, green_high_frequency_moments
from .util import orbital_occupations

class Solver(SolverCore):

    def __init__(self, beta, gf_struct, n_iw=1025, n_tau=10001, n_l=30, delta_interface = False):
        """
        Initialise the solver.

        Parameters
        ----------
        beta : scalar
               Inverse temperature.
        gf_struct : list of pairs [ (str,int), ...]
                    Structure of the Green's functions. It must be a
                    list of pairs, each containing the name of the
                    Green's function block and its linear size.
                    For example: ``[ ('up', 3), ('down', 3) ]``.
        n_iw : integer, optional
               Number of Matsubara frequencies used for the Green's functions.
        n_tau : integer, optional
               Number of imaginary time points used for the Green's functions.
        n_l : integer, optional
             Number of legendre polynomials to use in accumulations of the Green's functions.
        delta_interface: bool, optional
            Are Delta_tau and Delta_infty provided as input instead of G0_iw?
        """

        gf_struct = fix_gf_struct_type(gf_struct)

        # Initialise the core solver
        SolverCore.__init__(self, beta=beta, gf_struct=gf_struct,
                            n_iw=n_iw, n_tau=n_tau, n_l=n_l, delta_interface = delta_interface)

        mesh = MeshImFreq(beta = beta, S="Fermion", n_max = n_iw)
        self.Sigma_iw = BlockGf(mesh = mesh, gf_struct = gf_struct)
        self.Sigma_iw.zero()
        self.Sigma_iw_raw = None
        self.G_iw = self.Sigma_iw.copy()
        self.G_iw_raw = None
        self.G_iw_wo_moments = None
        self.gf_struct = gf_struct
        self.n_iw = n_iw
        self.n_tau = n_tau
        self.delta_interface = delta_interface
        self.G_moments = None
        self.Sigma_moments = None
        self.Sigma_Hartree = None

        # Improved Estimator Green's functions (iω): F^L, F^R, I
        self.G_iw_IE_FL = None
        self.G_iw_IE_FR = None
        self.G_iw_IE_I = None

        # Improved Estimator self-energies: Sigma^FG, Sigma^{IFG}
        self.Sigma_iw_IE_asym_L = None
        self.Sigma_iw_IE_asym_R = None
        self.Sigma_iw_IE_sym = None

    def solve(self, **params_kw):
        r"""
        Solve the impurity problem for a given G0_iw. If ``measure_G_tau``
        (default = ``True``), ``G_iw`` and ``Sigma_iw`` will be calculated and
        their tails fitted. In addition to the solver parameters, parameters to
        control the tail fitting can be provided.

        Moreover, the fundamental property :math:`G_{ij}(i \omega_n) =
        G_{ji}^*(- i \omega_n)` of the input G0_iw is enforced within C++, and
        a warning is printed if the property was not satisfied. Additionally, if
        ``measure_G_tau`` is set to ``True``, the property :math:`G_{ij}(\tau)=
        G_{ji}^*(\tau)` will be also ensured for the measured :math:`G(\tau)`.
	The difference between the original :math:`G(\tau)` and the hermitized
	:math:`G(\tau)` is stored in the object ``asymmetry_G_tau`` of the solver instance.


        Parameters
        ----------
        params_kw : dict {'param':value} that is passed to the core solver.
                     Two required :ref:`parameters <solve_parameters>` are
                        * `h_int` (:ref:`Operator object <triqslibs:operators>`): the local Hamiltonian of the impurity problem to be solved,
                        * `n_cycles` (int): number of measurements to be made.
        perform_post_proc : boolean, optional, default = ``True``
                            Should ``G_iw`` and ``Sigma_iw`` be calculated?
        perform_tail_fit : boolean, optional, default = ``False``
                           Should the tails of ``Sigma_iw`` and ``G_iw`` be fitted?
        fit_max_moment : integer, optional, default = 3
                         Highest moment to fit in the tail of ``Sigma_iw``.
        fit_known_moments : ``ndarray.shape[order, Sigma_iw[0].target_shape]``, optional, default = None
                            Known moments of Sigma_iw, given as an numpy ndarray
        fit_min_n : integer, optional, default = ``int(0.8 * self.n_iw)``
                    Index of ``iw`` from which to start fitting.
        fit_max_n : integer, optional, default = ``n_iw``
                    Index of ``iw`` to fit until.
        measure_G_tau_impr_est_v2 : bool, optional, default = False
                           Enable composite Green's function measurements for Improved Estimator.
                           Requires measure_G_tau = True.
                           If provided, post-processing will compute:

                           - G_iw_IE_FL, G_iw_IE_FR, G_iw_IE_I (Improved Estimator GFs: F^L, F^R, I)
                           - Sigma_iw_IE_asym_L, Sigma_iw_IE_asym_R (asymmetric self-energies: Sigma^FG)
                           - Sigma_iw_IE_sym (symmetric self-energy: Sigma^{IFG})

                           Reference: Fabian B. Kugler, Phys. Rev. B 105, 245132 (2022).
                           G_iw_IE_FL (F^L), G_iw_IE_FR (F^R), G_iw_IE_I (I): Eqs.(4, 5, A3, A4).
                           Sigma_iw_IE_asym_L, Sigma_iw_IE_asym_R (Sigma^FG): Eqs.(A7a, A7b).
                           Sigma_iw_IE_sym (Sigma^{IFG}): Eq.(A7i).
        """

        # -- Deprecation checks for measure parameters

        depr_params = dict(
            measure_g_tau='measure_G_tau',
            measure_g_l='measure_G_l',
            )

        for key in list(depr_params.keys()):
            if key in list(params_kw.keys()):
                print('WARNING: cthyb.solve parameter %s is deprecated use %s.' % \
                    (key, depr_params[key]))
                val = params_kw.pop(key)
                params_kw[depr_params[key]] = val

        # -- Tail post proc flags

        perform_post_proc = params_kw.pop("perform_post_proc", True)
        perform_tail_fit = params_kw.pop("perform_tail_fit", False)

        if perform_post_proc and perform_tail_fit:
            # If tail parameters provided for Sigma_iw fitting, use them, otherwise use defaults
            if not (("fit_min_n" in params_kw) or ("fit_max_n" in params_kw) or ("fit_max_w" in params_kw) or ("fit_min_w" in params_kw)):
                if mpi.is_master_node():
                    warning = ("!------------------------------------------------------------------------------------!\n"
                               "! WARNING: Using default high-frequency tail fitting parameters in the CTHYB solver. !\n"
                               "! You should check that the fitting range is suitable for your calculation!          !\n"
                               "!------------------------------------------------------------------------------------!")
                    print(warning)
            fit_min_n = params_kw.pop("fit_min_n", None)
            fit_max_n = params_kw.pop("fit_max_n", None)
            fit_min_w = params_kw.pop("fit_min_w", None)
            fit_max_w = params_kw.pop("fit_max_w", None)
            fit_max_moment = params_kw.pop("fit_max_moment", None)
            fit_known_moments = params_kw.pop("fit_known_moments", None)

        # Call the core solver's solve routine
        solve_status = SolverCore.solve(self, **params_kw)

        # Post-processing:
        # (only supported for G_tau, to permit compatibility with dft_tools)
        if perform_post_proc and (self.last_solve_parameters["measure_G_tau"] == True):

            if self.last_solve_parameters["measure_density_matrix"]:
                # we have the density matrix, so we will compute the high frequency
                # moments and orbital occupations

                self.orbital_occupations = orbital_occupations(self.density_matrix,
                                                               self.gf_struct,
                                                               self.h_loc_diagonalization
                                                               )

                h_int = self.last_solve_parameters['h_int']
                self.Sigma_moments = sigma_high_frequency_moments(self.density_matrix,
                                                 self.h_loc_diagonalization,
                                                 self.gf_struct,
                                                 h_int
                                                 )

                self.Sigma_Hartree = {bl: sigma_bl[0] for bl, sigma_bl in self.Sigma_moments.items()}

                self.G_moments = green_high_frequency_moments(self.density_matrix,
                                                         self.h_loc_diagonalization,
                                                         self.gf_struct,
                                                         self.h_loc
                                                         )

            # Fourier transform G_tau to obtain G_iw
            for bl, g in self.G_tau:
                bl_size = g.target_shape[0]
                if self.G_moments is None:
                    known_moments = make_zero_tail(g, 4)
                    known_moments[1] = np.eye(bl_size)
                else:
                    known_moments = self.G_moments[bl]
                self.G_iw[bl].set_from_fourier(g, known_moments)

            assert is_gf_hermitian(self.G_iw)
            self.G_iw_raw = self.G_iw.copy()

            if self.delta_interface:
                G0_iw = self.G_iw.copy()
                Delta_iw = make_gf_from_fourier(self.Delta_tau, self.n_iw)
                h_loc0_mat = block_matrix_from_op(self.h_loc0, self.gf_struct)
                for bl, bl_name in enumerate(self.Delta_tau.indices):
                    G0_iw[bl_name] << inverse(iOmega_n - Delta_iw[bl_name] - h_loc0_mat[bl])
            else:
                G0_iw = self.G0_iw

            # Solve Dyson's eq to obtain Sigma_iw and G_iw and fit the tail
            self.Sigma_iw = dyson(G0_iw=G0_iw, G_iw=self.G_iw)
            self.Sigma_iw_raw = self.Sigma_iw.copy()

            if perform_tail_fit:

                if fit_known_moments is None and self.Sigma_moments is not None:
                    fit_known_moments = self.Sigma_moments

                cthyb_tail_fit(
                    Sigma_iw=self.Sigma_iw,
                    fit_min_n = fit_min_n, fit_max_n = fit_max_n,
                    fit_min_w = fit_min_w, fit_max_w = fit_max_w,
                    fit_max_moment = fit_max_moment,
                    fit_known_moments = fit_known_moments,
                    )

                # Recompute G_iw with the fitted Sigma_iw
                self.G_iw = dyson(G0_iw=G0_iw, Sigma_iw=self.Sigma_iw)
            else:

                # Enforce 1/w behavior of G_iw in the tail fit window
                # and recompute Sigma_iw
                for bl, g in self.G_iw:
                    if self.G_moments is None:
                        tail = make_zero_tail(g, 2)
                        tail[1] = np.eye(g.target_shape[0])
                    else:
                        tail = self.G_moments[bl]

                    g.replace_by_tail_in_fit_window(tail)

                self.Sigma_iw = dyson(G0_iw=G0_iw, G_iw=self.G_iw)

        # Post-processing for measure_G_tau_impr_est_v2
        if perform_post_proc and self.last_solve_parameters.get("measure_G_tau_impr_est_v2"):

            # Error checking: measure_G_tau must be enabled
            if not self.last_solve_parameters.get("measure_G_tau", False):
                raise RuntimeError(
                    "Post-processing for measure_G_tau_impr_est_v2 requires measure_G_tau=True. "
                    "Please enable measure_G_tau in solve() parameters."
                )

            # Get the measured operator-decorated Green's functions from C++
            # Note: These are accessed with _debug suffix
            # Check that measurements exist
            if self.G_tau_with_O1_debug is None or self.G_tau_with_O2_debug is None or self.G_tau_with_O1_O2_debug is None:
                raise RuntimeError(
                    "measure_G_tau_impr_est_v2 measurements not found. "
                    "Ensure the measurement was enabled correctly."
                )

            # Compute Improved Estimator Green's functions in imaginary time
            # Remove H_loc0 contribution from the measured operator-decorated GFs
            # In cthyb, we measured the following:
            # G_with_O1    = <[H_loc, c](t),         c† (0)>
            # G_with_O2    = <        c (t), [H_loc, c†](0)>
            # G_with_O1_O2 = <[H_loc, c](t), [H_loc, c†](0)>
            #
            # We want q = [c, H_int], q† = [H_int, c†] with H_loc = H_loc0 + H_int
            # [H_loc, c ] = [H_loc0, c ] + [H_int, c ] = - e0_loc * c  - q
            # [H_loc, c†] = [H_loc0, c†] + [H_int, c†] = + c† * e0_loc + q†
            #
            # The composite Green functions are:
            # G_IE_1  = <q(t), c†(0)> = -G_with_O1 - e0_loc @ G
            # G_IE_2  = <c(t), q†(0)> = +G_with_O2 - G @ e0_loc
            # G_IE_12 = <q(t), q†(0)> = -G_with_O1_O2 - G_IE_1 @ e0_loc - e0_loc @ G_IE_2 - e0_loc @ G @ e0_loc

            # Extract e0_loc from h_loc0 using block_matrix_from_op
            # This converts the operator representation to block matrix form
            e0_loc_dict = block_matrix_from_op(self.h_loc0, self.gf_struct)

            G_tau_IE_FL = -self.G_tau_with_O1_debug.copy()
            G_tau_IE_FR =  self.G_tau_with_O2_debug.copy()
            G_tau_IE_I  = -self.G_tau_with_O1_O2_debug.copy()

            for bl_idx, (bl, g) in enumerate(self.G_tau):
                e0_loc = e0_loc_dict[bl_idx]
                G_tau_IE_FL[bl] -= e0_loc @ g
                G_tau_IE_FR[bl] -= g @ e0_loc
                G_tau_IE_I[bl]  -= G_tau_IE_FL[bl] @ e0_loc
                G_tau_IE_I[bl]  -= e0_loc @ G_tau_IE_FR[bl]
                G_tau_IE_I[bl]  -= e0_loc @ g @ e0_loc

            # Compute Improved Estimator Green's functions in frequency domain
            # by Fourier transforming the imaginary time versions (no known moments)
            # G_iw_wo_moments: G_tau Fourier-transformed without moments, for consistency
            self.G_iw_wo_moments = self.G_iw.copy()
            self.G_iw_IE_FL = self.G_iw.copy()
            self.G_iw_IE_FR = self.G_iw.copy()
            self.G_iw_IE_I  = self.G_iw.copy()

            for bl, g in self.G_iw:
                known_moments = make_zero_tail(g, 1)
                self.G_iw_wo_moments[bl].set_from_fourier(self.G_tau[bl], known_moments)
                self.G_iw_IE_FL[bl].set_from_fourier(G_tau_IE_FL[bl], known_moments)
                self.G_iw_IE_FR[bl].set_from_fourier(G_tau_IE_FR[bl], known_moments)
                self.G_iw_IE_I[bl].set_from_fourier(G_tau_IE_I[bl], known_moments)

            # Compute Improved Estimator self-energies [Eqs.(A7a, A7b, A7i)]
            # Use G_iw_wo_moments for consistency: all GFs Fourier-transformed without moments
            self.Sigma_iw_IE_asym_L = self.G_iw_IE_FL * inverse(self.G_iw_wo_moments)
            self.Sigma_iw_IE_asym_R = inverse(self.G_iw_wo_moments) * self.G_iw_IE_FR
            self.Sigma_iw_IE_sym = self.G_iw_IE_I - self.G_iw_IE_FL * inverse(self.G_iw_wo_moments) * self.G_iw_IE_FR

            # Add Hartree self-energy if available (from density matrix measurement)
            if self.Sigma_Hartree is not None:
                for bl, g in self.Sigma_iw_IE_sym:
                    g += self.Sigma_Hartree[bl]
            else:
                if mpi.is_master_node():
                    print("WARNING: measure_density_matrix=False, Hartree self-energy not added to Sigma_iw_IE_sym.")

        return solve_status
