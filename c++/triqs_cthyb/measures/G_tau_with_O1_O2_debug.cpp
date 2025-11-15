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

#include "./G_tau_with_O1_O2_debug.hpp"
#include <set>
#include <map>
#include <triqs/operators/many_body_operator.hpp>

namespace triqs_cthyb {

  using namespace triqs::gfs;
  using namespace triqs::mesh;

  measure_G_tau_with_O1_O2_debug::measure_G_tau_with_O1_O2_debug(qmc_data const &data, int n_tau, gf_struct_t const &gf_struct,
                                                                 many_body_op_t const &op1, many_body_op_t const &op2,
                                                                 container_set_t &results)
     : data(data), average_sign(0) {
    // Use debug container fields
    results.G_tau_with_O1_O2_debug = block_gf<imtime, G_target_t>({data.config.beta(), Fermion, n_tau}, gf_struct);
    G_tau_with_O1_O2.rebind(*results.G_tau_with_O1_O2_debug);
    G_tau_with_O1_O2() = 0.0;

    results.G_tau_with_O1_debug = block_gf<imtime, G_target_t>({data.config.beta(), Fermion, n_tau}, gf_struct);
    G_tau_with_O1.rebind(*results.G_tau_with_O1_debug);
    G_tau_with_O1() = 0.0;

    results.G_tau_with_O2_debug = block_gf<imtime, G_target_t>({data.config.beta(), Fermion, n_tau}, gf_struct);
    G_tau_with_O2.rebind(*results.G_tau_with_O2_debug);
    G_tau_with_O2() = 0.0;

    // Attach operators to impurity trace as auxiliary operators
    op1_d = data.imp_trace.attach_aux_operator(op1);
    op2_d = data.imp_trace.attach_aux_operator(op2);

    // Pre-construct commutator operators for all (block, inner_index) pairs
    // This avoids repeated construction during accumulate()
    int block_idx = 0;
    for (auto const &[block_name, block_size] : gf_struct) {
      for (int inner = 0; inner < block_size; ++inner) {
        auto key = std::make_pair(block_idx, inner);

        // Construct fundamental operators using block name and inner index
        // Same approach as in solver_core.cpp:182
        many_body_op_t c_op = triqs::operators::c<h_scalar_t>(block_name, inner);         // c operator
        many_body_op_t cdag_op = triqs::operators::c_dag<h_scalar_t>(block_name, inner);  // c† operator

        // Construct and attach [O1, c_β] commutator
        auto comm_O1 = op1 * c_op - c_op * op1;
        comm_O1_c[key] = data.imp_trace.attach_aux_operator(comm_O1);

        // Construct and attach [O2, c_β†] commutator
        auto comm_O2 = op2 * cdag_op - cdag_op * op2;
        comm_O2_cdag[key] = data.imp_trace.attach_aux_operator(comm_O2);
      }
      block_idx++;
    }
  }

  void measure_G_tau_with_O1_O2_debug::accumulate(mc_weight_t s) {
    s *= data.atomic_reweighting;
    average_sign += s;

    // Compute baseline trace WITHOUT inserted operators (once per configuration)
    auto [bare_atomic_weight, bare_atomic_reweighting] = data.imp_trace.compute();
    const auto baseline_trace = bare_atomic_weight * bare_atomic_reweighting;

    // OPTIMIZATION: Pre-compute single-operator traces to avoid redundant computation
    // Cache for storing trace ratios with single operator insertions
    std::map<time_pt, h_scalar_t> op1_trace_cache;
    std::map<time_pt, h_scalar_t> op2_trace_cache;
    std::map<std::pair<time_pt, time_pt>, h_scalar_t> op12_trace_cache;

    // Reusable map for operator replacements (avoids repeated allocations)
    configuration::oplist_t updated_ops;

    // Loop over all orbital blocks
    for (auto block_idx : range(G_tau_with_O1_O2.size())) {
      auto &det = data.dets[block_idx];
      int det_size = det.size();

      // PHASE 2: Pre-compute O1 traces using try_replace
      // Replace c_β(τ_y) with [O1, c_β] to compute commutator directly
      for (int j = 0; j < det_size; ++j) {
        auto const &y = det.get_y(j);  // y = {tau_y, inner_index}
        auto tau_y = y.first;
        auto key = std::make_pair(block_idx, y.second);

        // Replace c(tau_y) with [O1, c] commutator
        updated_ops.clear();
        updated_ops[tau_y] = comm_O1_c[key];

        data.imp_trace.try_replace(updated_ops);
        auto [w, rw] = data.imp_trace.compute();
        op1_trace_cache[tau_y] = w * rw;
        data.imp_trace.cancel_replace();
      }

      // PHASE 3: Pre-compute O2 traces using try_replace
      // Replace c_α†(τ_x) with [O2, c_α†] to compute commutator directly
      for (int i = 0; i < det_size; ++i) {
        auto const &x = det.get_x(i);  // x = {tau_x, inner_index}
        auto tau_x = x.first;
        auto key = std::make_pair(block_idx, x.second);

        // Replace c†(tau_x) with [O2, c†] commutator
        updated_ops.clear();
        updated_ops[tau_x] = comm_O2_cdag[key];

        data.imp_trace.try_replace(updated_ops);
        auto [w, rw] = data.imp_trace.compute();
        op2_trace_cache[tau_x] = w * rw;
        data.imp_trace.cancel_replace();
      }

      // PHASE 4: Pre-compute O1+O2 traces using joint try_replace
      // Replace both c_β(τ_y) with [O1, c_β] AND c_α†(τ_x) with [O2, c_α†]
      for (int j = 0; j < det_size; ++j) {
        auto const &y = det.get_y(j);
        auto tau_y = y.first;
        auto key_y = std::make_pair(block_idx, y.second);

        // Insert O1 commutator once for this y (outer loop)
        updated_ops.clear();
        updated_ops[tau_y] = comm_O1_c[key_y];

        for (int i = 0; i < det_size; ++i) {
          auto const &x = det.get_x(i);
          auto tau_x = x.first;
          auto key_x = std::make_pair(block_idx, x.second);

          // Insert O2 commutator for this x
          updated_ops[tau_x] = comm_O2_cdag[key_x];

          data.imp_trace.try_replace(updated_ops);
          auto [w, rw] = data.imp_trace.compute();
          op12_trace_cache[{tau_y, tau_x}] = w * rw;
          data.imp_trace.cancel_replace();

          // Remove O2 commutator for next iteration
          updated_ops.erase(tau_x);
        }
      }

      // PHASE 5: Iterate over all (c†, c) pairs in the determinant for this block
      // Computing: ⟨O1 c_β(τ_y) O2 c_α†(τ_x)⟩ - ⟨c_β(τ_y) O1 O2 c_α†(τ_x)⟩ - ⟨O1 c_β(τ_y) c_α†(τ_x) O2⟩ + ⟨c_β(τ_y) O1 c_α†(τ_x) O2⟩
      foreach (data.dets[block_idx], [this, s, block_idx, baseline_trace,
        &op1_trace_cache, &op2_trace_cache, &op12_trace_cache](op_t const &x, op_t const &y, det_scalar_t M) {

        // x = c†(τ_x, α) creation operator
        // y = c(τ_y, β) annihilation operator
        // M = [Δ⁻¹]_{βα} (inverse hybridization matrix element)

        // Case 1: measure ⟨[O1, c_β](τ_y) [O2, c_α†](τ_x)⟩
        {
          // Lookup pre-computed trace from cache
          auto modified_trace = op12_trace_cache[{y.first, x.first}];

          // Check if this is a valid cache entry (not a sentinel from failed insertion)
          if (modified_trace != h_scalar_t(0)) {
            auto trace_ratio = modified_trace / baseline_trace;

            // Handle fermionic antiperiodicity
            auto sign = (y.first >= x.first ? 1.0 : -1.0);

            // Weight = MC_weight × sign × trace_ratio × det_inverse
            auto val = s * sign * trace_ratio * M;

            // Accumulate to histogram at time difference τ' - τ
            double dtau = double(y.first - x.first);
            this->G_tau_with_O1_O2[block_idx][closest_mesh_pt(dtau)](y.second, x.second) += val;
          }
        }

        // Case 2: O1 inserted AFTER c (at τ_y+ε) - for G_tau_with_O1 [OPTIMIZED: CACHED]
        // Measures: ⟨O1 c_β(τ_y) c_α†(τ_x)⟩ - ⟨c_β(τ_y) O1 c_α†(τ_x)⟩
        {
          // Lookup pre-computed trace from cache
          auto modified_trace = op1_trace_cache[y.first];

          // Check if this is a valid cache entry (not a sentinel from failed insertion)
          if (modified_trace != h_scalar_t(0)) {
            auto trace_ratio = modified_trace / baseline_trace;

            // Handle fermionic antiperiodicity
            auto sign = (y.first >= x.first ? 1.0 : -1.0);

            // Weight = MC_weight × sign × trace_ratio × det_inverse
            auto val = s * sign * trace_ratio * M;

            // Accumulate to G_tau_with_O1
            double dtau = double(y.first - x.first);
            this->G_tau_with_O1[block_idx][closest_mesh_pt(dtau)](y.second, x.second) += val;
          }
        }

        // Case 3: O2 inserted AFTER c† (at τ_x+ε) - for G_tau_with_O2 [OPTIMIZED: CACHED]
        // Measures: ⟨c_β(τ_y) O2 c_α†(τ_x)⟩ - ⟨c_β(τ_y) c_α†(τ_x) O2⟩
        {
          // Lookup pre-computed trace from cache
          auto modified_trace = op2_trace_cache[x.first];

          // Check if this is a valid cache entry
          if (modified_trace != h_scalar_t(0)) {
            auto trace_ratio = modified_trace / baseline_trace;

            // Handle fermionic antiperiodicity
            auto sign = (y.first >= x.first ? 1.0 : -1.0);

            // Weight = MC_weight × sign × trace_ratio × det_inverse
            auto val = s * sign * trace_ratio * M;

            // Accumulate to G_tau_with_O2
            double dtau = double(y.first - x.first);
            this->G_tau_with_O2[block_idx][closest_mesh_pt(dtau)](y.second, x.second) += val;
          }
        }
      });
    }
  }

  void measure_G_tau_with_O1_O2_debug::collect_results(mpi::communicator const &c) {

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
