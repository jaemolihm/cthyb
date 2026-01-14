/*******************************************************************************
 *
 * TRIQS: a Toolbox for Research in Interacting Quantum Systems
 *
 * Copyright (C) 2018, The Simons Foundation
 * Author: H. U.R. Strand
 *
 * TRIQS is free software: you can redistribute it and/or modify it under the
 * terms of the GNU General Public License as published by the Free Software
 * Foundation, either version 3 of the License, or (at your option) any later
 * version.
 *
 * TRIQS is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 * FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
 * details.
 *
 * You should have received a copy of the GNU General Public License along with
 * TRIQS. If not, see <http://www.gnu.org/licenses/>.
 *
 ******************************************************************************/

#include "./G_tau_impr_est.hpp"

namespace triqs_cthyb {

  using namespace triqs::gfs;
  using namespace triqs::mesh;

  measure_G_tau_impr_est::measure_G_tau_impr_est(qmc_data const &data, int n_tau, gf_struct_t const &gf_struct,
                                                 many_body_op_t const &h_op,
                                                 container_set_t &results)
     : data(data), average_sign(0) {
    results.G_tau_with_O1_O2_accum = block_gf<imtime, G_target_t>({data.config.beta(), Fermion, n_tau}, gf_struct);
    G_tau_with_O1_O2.rebind(*results.G_tau_with_O1_O2_accum);
    G_tau_with_O1_O2() = 0.0;

    results.G_tau_with_O1_accum = block_gf<imtime, G_target_t>({data.config.beta(), Fermion, n_tau}, gf_struct);
    G_tau_with_O1.rebind(*results.G_tau_with_O1_accum);
    G_tau_with_O1() = 0.0;

    results.G_tau_with_O2_accum = block_gf<imtime, G_target_t>({data.config.beta(), Fermion, n_tau}, gf_struct);
    G_tau_with_O2.rebind(*results.G_tau_with_O2_accum);
    G_tau_with_O2() = 0.0;

    // Attach h_op to impurity trace as auxiliary operator
    h_op_d = data.imp_trace.attach_aux_operator(h_op);
  }

  void measure_G_tau_impr_est::accumulate(mc_weight_t s) {
    s *= data.atomic_reweighting;
    average_sign += s;

    // Compute baseline trace WITHOUT inserted operators (once per configuration)
    auto [bare_atomic_weight, bare_atomic_reweighting] = data.imp_trace.compute();
    const auto baseline_trace = bare_atomic_weight * bare_atomic_reweighting;

    // Loop over all orbital blocks
    for (auto block_idx : range(G_tau_with_O1_O2.size())) {

      // Iterate over all (c†, c) pairs in the determinant for this block
      // Computing:  ⟨H_loc c_β(τ_y) H_loc c†_α(τ_x)⟩ - ⟨c_β(τ_y) H_loc H_loc c†_α(τ_x)⟩
      //           - ⟨H_loc c_β(τ_y) c†_α(τ_x) H_loc⟩ + ⟨c_β(τ_y) H_loc c†_α(τ_x) H_loc⟩
      foreach (data.dets[block_idx], [this, s, block_idx, baseline_trace](op_t const &x, op_t const &y, det_scalar_t M) {

        // x = c†(τ_x, α) creation operator
        // y = c(τ_y, β) annihilation operator
        // M = [Δ⁻¹]_{βα} (inverse hybridization matrix element)

        // Case 1: H_loc inserted AFTER c (at τ_y+ε), H_loc inserted AFTER c† (at τ_x+ε)
        // Measures: ⟨H_loc c_β(τ_y) H_loc c†_α(τ_x)⟩
        auto tau_h_after_y = y.first + time_pt{1, data.config.beta()};
        auto tau_h_after_x = x.first + time_pt{1, data.config.beta()};

        try {
          // Insert both H_loc operators into impurity trace
          data.imp_trace.try_insert(tau_h_after_y, h_op_d);
          data.imp_trace.try_insert(tau_h_after_x, h_op_d);

          // Compute trace with both H_loc operators inserted
          auto [w, rw] = data.imp_trace.compute();
          auto trace_ratio = (w * rw) / baseline_trace;

          // Handle fermionic antiperiodicity
          auto sign = (y.first >= x.first ? 1.0 : -1.0);

          // Weight = MC_weight × sign × trace_ratio × det_inverse
          auto val = s * sign * trace_ratio * M;

          // Accumulate to histogram at time difference τ' - τ
          double dtau = double(y.first - x.first);
          this->G_tau_with_O1_O2[block_idx][closest_mesh_pt(dtau)](y.second, x.second) += val;

        } catch (rbt_insert_error const &) {
          // Insertion failed (extremely rare: both H_loc insertions at nearly identical times)
          // Just skip this contribution
        }

        // Clean up: remove insertions for next iteration
        data.imp_trace.cancel_insert();

        // Case 2: H_loc inserted BEFORE c (at τ_y-ε), H_loc inserted AFTER c† (at τ_x+ε)
        // Measures: ⟨c_β(τ_y) H_loc H_loc c†_α(τ_x)⟩
        auto tau_h_before_y = y.first - time_pt{1, data.config.beta()};
        auto tau_h_after_x_2 = x.first + time_pt{1, data.config.beta()};

        try {
          // Insert both H_loc operators into impurity trace
          data.imp_trace.try_insert(tau_h_before_y, h_op_d);
          data.imp_trace.try_insert(tau_h_after_x_2, h_op_d);

          // Compute trace with both H_loc operators inserted
          auto [w, rw] = data.imp_trace.compute();
          auto trace_ratio = (w * rw) / baseline_trace;

          // Handle fermionic antiperiodicity
          auto sign = (y.first >= x.first ? 1.0 : -1.0);

          // Weight = MC_weight × sign × trace_ratio × det_inverse
          auto val = s * sign * trace_ratio * M;

          // Accumulate to histogram at time difference τ' - τ (with SUBTRACTION)
          double dtau = double(y.first - x.first);
          this->G_tau_with_O1_O2[block_idx][closest_mesh_pt(dtau)](y.second, x.second) -= val;

        } catch (rbt_insert_error const &) {
          // Insertion failed (extremely rare: both H_loc insertions at nearly identical times)
          // Just skip this contribution
        }

        // Clean up: remove insertions for next iteration
        data.imp_trace.cancel_insert();

        // Case 3: H_loc inserted AFTER c (at τ_y+ε), H_loc inserted BEFORE c† (at τ_x-ε)
        // Measures: ⟨H_loc c_β(τ_y) c†_α(τ_x) H_loc⟩
        auto tau_h_after_y_3 = y.first + time_pt{1, data.config.beta()};
        auto tau_h_before_x = x.first - time_pt{1, data.config.beta()};

        try {
          // Insert both H_loc operators into impurity trace
          data.imp_trace.try_insert(tau_h_after_y_3, h_op_d);
          data.imp_trace.try_insert(tau_h_before_x, h_op_d);

          // Compute trace with both H_loc operators inserted
          auto [w, rw] = data.imp_trace.compute();
          auto trace_ratio = (w * rw) / baseline_trace;

          // Handle fermionic antiperiodicity
          auto sign = (y.first >= x.first ? 1.0 : -1.0);

          // Weight = MC_weight × sign × trace_ratio × det_inverse
          auto val = s * sign * trace_ratio * M;

          // Accumulate to histogram at time difference τ' - τ (with SUBTRACTION)
          double dtau = double(y.first - x.first);
          this->G_tau_with_O1_O2[block_idx][closest_mesh_pt(dtau)](y.second, x.second) -= val;

        } catch (rbt_insert_error const &) {
          // Insertion failed (extremely rare: both H_loc insertions at nearly identical times)
          // Just skip this contribution
        }

        // Clean up: remove insertions for next iteration
        data.imp_trace.cancel_insert();

        // Case 4: H_loc inserted BEFORE c (at τ_y-ε), H_loc inserted BEFORE c† (at τ_x-ε)
        // Measures: ⟨c_β(τ_y) H_loc c†_α(τ_x) H_loc⟩
        auto tau_h_before_y_4 = y.first - time_pt{1, data.config.beta()};
        auto tau_h_before_x_4 = x.first - time_pt{1, data.config.beta()};

        try {
          // Insert both H_loc operators into impurity trace
          data.imp_trace.try_insert(tau_h_before_y_4, h_op_d);
          data.imp_trace.try_insert(tau_h_before_x_4, h_op_d);

          // Compute trace with both H_loc operators inserted
          auto [w, rw] = data.imp_trace.compute();
          auto trace_ratio = (w * rw) / baseline_trace;

          // Handle fermionic antiperiodicity
          auto sign = (y.first >= x.first ? 1.0 : -1.0);

          // Weight = MC_weight × sign × trace_ratio × det_inverse
          auto val = s * sign * trace_ratio * M;

          // Accumulate to histogram at time difference τ' - τ (with ADDITION)
          double dtau = double(y.first - x.first);
          this->G_tau_with_O1_O2[block_idx][closest_mesh_pt(dtau)](y.second, x.second) += val;

        } catch (rbt_insert_error const &) {
          // Insertion failed (extremely rare: tau_O1 == tau_O2 within machine precision)
          // Just skip this contribution
        }

        // Clean up: remove insertions for next iteration
        data.imp_trace.cancel_insert();

        // Case 5: H_loc inserted AFTER c (at τ_y+ε) - for G_tau_with_O1
        // Measures: ⟨H_loc c_β(τ_y) c†_α(τ_x)⟩
        auto tau_h_only_after_y = y.first + time_pt{1, data.config.beta()};

        try {
          // Insert only H_loc after c
          data.imp_trace.try_insert(tau_h_only_after_y, h_op_d);

          // Compute trace with H_loc inserted
          auto [w, rw] = data.imp_trace.compute();
          auto trace_ratio = (w * rw) / baseline_trace;

          // Handle fermionic antiperiodicity
          auto sign = (y.first >= x.first ? 1.0 : -1.0);

          // Weight = MC_weight × sign × trace_ratio × det_inverse
          auto val = s * sign * trace_ratio * M;

          // Accumulate to G_tau_with_O1
          double dtau = double(y.first - x.first);
          this->G_tau_with_O1[block_idx][closest_mesh_pt(dtau)](y.second, x.second) += val;

        } catch (rbt_insert_error const &) {
          // Insertion failed
        }

        // Clean up
        data.imp_trace.cancel_insert();

        // Case 6: H_loc inserted BEFORE c (at τ_y-ε) - for G_tau_with_O1
        // Measures: ⟨c_β(τ_y) H_loc c†_α(τ_x)⟩
        auto tau_h_only_before_y = y.first - time_pt{1, data.config.beta()};

        try {
          // Insert only H_loc before c
          data.imp_trace.try_insert(tau_h_only_before_y, h_op_d);

          // Compute trace with H_loc inserted
          auto [w, rw] = data.imp_trace.compute();
          auto trace_ratio = (w * rw) / baseline_trace;

          // Handle fermionic antiperiodicity
          auto sign = (y.first >= x.first ? 1.0 : -1.0);

          // Weight = MC_weight × sign × trace_ratio × det_inverse
          auto val = s * sign * trace_ratio * M;

          // Accumulate to G_tau_with_O1 (with SUBTRACTION)
          double dtau = double(y.first - x.first);
          this->G_tau_with_O1[block_idx][closest_mesh_pt(dtau)](y.second, x.second) -= val;

        } catch (rbt_insert_error const &) {
          // Insertion failed
        }

        // Clean up
        data.imp_trace.cancel_insert();

        // Case 7: H_loc inserted AFTER c† (at τ_x+ε) - for G_tau_with_O2
        // Measures: ⟨c_β(τ_y) H_loc c†_α(τ_x)⟩
        auto tau_h_only_after_x = x.first + time_pt{1, data.config.beta()};

        try {
          // Insert only H_loc after c†
          data.imp_trace.try_insert(tau_h_only_after_x, h_op_d);

          // Compute trace with H_loc inserted
          auto [w, rw] = data.imp_trace.compute();
          auto trace_ratio = (w * rw) / baseline_trace;

          // Handle fermionic antiperiodicity
          auto sign = (y.first >= x.first ? 1.0 : -1.0);

          // Weight = MC_weight × sign × trace_ratio × det_inverse
          auto val = s * sign * trace_ratio * M;

          // Accumulate to G_tau_with_O2
          double dtau = double(y.first - x.first);
          this->G_tau_with_O2[block_idx][closest_mesh_pt(dtau)](y.second, x.second) += val;

        } catch (rbt_insert_error const &) {
          // Insertion failed
        }

        // Clean up
        data.imp_trace.cancel_insert();

        // Case 8: H_loc inserted BEFORE c† (at τ_x-ε) - for G_tau_with_O2
        // Measures: ⟨c_β(τ_y) c†_α(τ_x) H_loc⟩
        auto tau_h_only_before_x = x.first - time_pt{1, data.config.beta()};

        try {
          // Insert only H_loc before c†
          data.imp_trace.try_insert(tau_h_only_before_x, h_op_d);

          // Compute trace with H_loc inserted
          auto [w, rw] = data.imp_trace.compute();
          auto trace_ratio = (w * rw) / baseline_trace;

          // Handle fermionic antiperiodicity
          auto sign = (y.first >= x.first ? 1.0 : -1.0);

          // Weight = MC_weight × sign × trace_ratio × det_inverse
          auto val = s * sign * trace_ratio * M;

          // Accumulate to G_tau_with_O2 (with SUBTRACTION)
          double dtau = double(y.first - x.first);
          this->G_tau_with_O2[block_idx][closest_mesh_pt(dtau)](y.second, x.second) -= val;

        } catch (rbt_insert_error const &) {
          // Insertion failed
        }

        // Clean up
        data.imp_trace.cancel_insert();
      });
    }
  }

  void measure_G_tau_impr_est::collect_results(mpi::communicator const &c) {

    // MPI reduction
    G_tau_with_O1_O2 = mpi::all_reduce(G_tau_with_O1_O2, c);
    G_tau_with_O1 = mpi::all_reduce(G_tau_with_O1, c);
    G_tau_with_O2 = mpi::all_reduce(G_tau_with_O2, c);
    average_sign = mpi::all_reduce(average_sign, c);

    // Normalize G_tau_with_O1_O2
    for (auto &G_block : G_tau_with_O1_O2) {
      double beta = G_block.mesh().beta();

      // Normalize by average sign, beta, and bin width
      G_block /= -real(average_sign) * beta * G_block.mesh().delta();

      // Multiply first and last bins by 2 (half-sized bins)
      int last = G_block.mesh().size() - 1;
      G_block[0] *= 2;
      G_block[last] *= 2;

      // Enforce fermionic discontinuity: G(0⁻) - G(0⁺) = -1
      // NOTE: Commented out - not applicable for operator-decorated Green's functions
      // G_block[0] = 0.5 * matrix_t(G_block[0] - 1 - G_block[last]);
      // G_block[last] = -1 - G_block[0];
    }

    // Normalize G_tau_with_O1
    for (auto &G_block : G_tau_with_O1) {
      double beta = G_block.mesh().beta();
      G_block /= -real(average_sign) * beta * G_block.mesh().delta();
      int last = G_block.mesh().size() - 1;
      G_block[0] *= 2;
      G_block[last] *= 2;
    }

    // Normalize G_tau_with_O2
    for (auto &G_block : G_tau_with_O2) {
      double beta = G_block.mesh().beta();
      G_block /= -real(average_sign) * beta * G_block.mesh().delta();
      int last = G_block.mesh().size() - 1;
      G_block[0] *= 2;
      G_block[last] *= 2;
    }
  }

} // namespace triqs_cthyb
