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

#include "./G_tau_impr_est_v2.hpp"
#include <set>
#include <map>
#include <triqs/operators/many_body_operator.hpp>

namespace triqs_cthyb {

  using namespace triqs::gfs;
  using namespace triqs::mesh;

  measure_G_tau_impr_est_v2::measure_G_tau_impr_est_v2(qmc_data const &data, int n_tau, gf_struct_t const &gf_struct,
                                                       many_body_op_t const &h_op,
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

    // Pre-construct H_loc commutator operators for all (block, inner_index) pairs
    // This avoids repeated construction during accumulate()
    int block_idx = 0;
    for (auto const &[block_name, block_size] : gf_struct) {
      for (int inner = 0; inner < block_size; ++inner) {
        auto key = std::make_pair(block_idx, inner);

        // Construct fundamental operators using block name and inner index
        // Same approach as in solver_core.cpp:182
        many_body_op_t c_op = triqs::operators::c<h_scalar_t>(block_name, inner);         // c operator
        many_body_op_t cdag_op = triqs::operators::c_dag<h_scalar_t>(block_name, inner);  // c† operator

        // Construct and attach [H_loc, c_β] commutator
        auto comm_H_c_op = h_op * c_op - c_op * h_op;
        comm_H_c[key] = data.imp_trace.attach_aux_operator(comm_H_c_op);

        // Construct and attach [H_loc, c_β†] commutator
        auto comm_H_cdag_op = h_op * cdag_op - cdag_op * h_op;
        comm_H_cdag[key] = data.imp_trace.attach_aux_operator(comm_H_cdag_op);
      }
      block_idx++;
    }
  }

  void measure_G_tau_impr_est_v2::accumulate(mc_weight_t s) {
    s *= data.atomic_reweighting;
    average_sign += s;

    // Compute baseline trace WITHOUT inserted operators (once per configuration)
    auto [bare_atomic_weight, bare_atomic_reweighting] = data.imp_trace.compute();
    const auto baseline_trace = bare_atomic_weight * bare_atomic_reweighting;

    // OPTIMIZATION: Pre-compute single-operator traces to avoid redundant computation
    // Cache for storing trace ratios with H_loc commutator insertions
    std::map<time_pt, h_scalar_t> h_comm_c_trace_cache;       // [H_loc, c] traces
    std::map<time_pt, h_scalar_t> h_comm_cdag_trace_cache;    // [H_loc, c†] traces
    std::map<std::pair<time_pt, time_pt>, h_scalar_t> h_comm_both_trace_cache;  // both commutators

    // Reusable map for operator replacements (avoids repeated allocations)
    configuration::oplist_t updated_ops;

    // Loop over all orbital blocks
    for (auto block_idx : range(G_tau_with_O1_O2.size())) {
      auto &det = data.dets[block_idx];
      int det_size = det.size();

      // Pre-compute [H_loc, c] traces using try_replace
      // Replace c_β(τ_y) with [H_loc, c_β] to compute commutator directly
      for (int j = 0; j < det_size; ++j) {
        auto const &y = det.get_y(j);  // y = {tau_y, inner_index}
        auto tau_y = y.first;
        auto key = std::make_pair(block_idx, y.second);

        // Replace c(tau_y) with [H_loc, c] commutator
        updated_ops.clear();
        updated_ops[tau_y] = comm_H_c[key];

        data.imp_trace.try_replace(updated_ops);
        auto [w, rw] = data.imp_trace.compute();
        h_comm_c_trace_cache[tau_y] = w * rw;
        data.imp_trace.cancel_replace();
      }

      // Pre-compute [H_loc, c†] traces using try_replace
      // Replace c_α†(τ_x) with [H_loc, c_α†] to compute commutator directly
      for (int i = 0; i < det_size; ++i) {
        auto const &x = det.get_x(i);  // x = {tau_x, inner_index}
        auto tau_x = x.first;
        auto key = std::make_pair(block_idx, x.second);

        // Replace c†(tau_x) with [H_loc, c†] commutator
        updated_ops.clear();
        updated_ops[tau_x] = comm_H_cdag[key];

        data.imp_trace.try_replace(updated_ops);
        auto [w, rw] = data.imp_trace.compute();
        h_comm_cdag_trace_cache[tau_x] = w * rw;
        data.imp_trace.cancel_replace();
      }

      // Pre-compute traces with both [H_loc, c] and [H_loc, c†] using joint try_replace
      // Replace both c_β(τ_y) with [H_loc, c_β] AND c_α†(τ_x) with [H_loc, c_α†]
      for (int j = 0; j < det_size; ++j) {
        auto const &y = det.get_y(j);
        auto tau_y = y.first;
        auto key_y = std::make_pair(block_idx, y.second);

        // Insert [H_loc, c] commutator once for this y (outer loop)
        updated_ops.clear();
        updated_ops[tau_y] = comm_H_c[key_y];

        for (int i = 0; i < det_size; ++i) {
          auto const &x = det.get_x(i);
          auto tau_x = x.first;
          auto key_x = std::make_pair(block_idx, x.second);

          // Insert [H_loc, c†] commutator for this x
          updated_ops[tau_x] = comm_H_cdag[key_x];

          data.imp_trace.try_replace(updated_ops);
          auto [w, rw] = data.imp_trace.compute();
          h_comm_both_trace_cache[{tau_y, tau_x}] = w * rw;
          data.imp_trace.cancel_replace();

          // Remove [H_loc, c†] commutator for next iteration
          updated_ops.erase(tau_x);
        }
      }

      // Iterate over all (c†, c) pairs in the determinant for this block
      foreach (data.dets[block_idx], [this, s, block_idx, baseline_trace,
        &h_comm_c_trace_cache, &h_comm_cdag_trace_cache, &h_comm_both_trace_cache](op_t const &x, op_t const &y, det_scalar_t M) {

        // x = c†(τ_x, α) creation operator
        // y = c(τ_y, β) annihilation operator
        // M = [Δ⁻¹]_{βα} (inverse hybridization matrix element)

        // Case 1: measure ⟨[H_loc, c_β](τ_y) [H_loc, c_α†](τ_x)⟩
        {
          // Lookup pre-computed trace from cache
          auto modified_trace = h_comm_both_trace_cache[{y.first, x.first}];

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

        // Case 2: measure ⟨[H_loc, c_β(τ_y)] c_α†(τ_x)⟩
        {
          // Lookup pre-computed trace from cache
          auto modified_trace = h_comm_c_trace_cache[y.first];

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

        // Case 3: measure ⟨c_β(τ_y) [H_loc, c_α†(τ_x)]⟩
        {
          // Lookup pre-computed trace from cache
          auto modified_trace = h_comm_cdag_trace_cache[x.first];

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

  void measure_G_tau_impr_est_v2::collect_results(mpi::communicator const &c) {

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
