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
#pragma once

#include <triqs/gfs.hpp>
#include <triqs/mesh.hpp>

#include "../qmc_data.hpp"
#include "../container_set.hpp"

namespace triqs_cthyb {

  using namespace triqs::gfs;
  using namespace triqs::mesh;

  // V2 VERSION: Measure imaginary time Green's function with operator insertions (all blocks)
  // This version implements single-operator trace caching optimization.
  // For performance comparison with the original measure_G_tau_impr_est.
  class measure_G_tau_impr_est_v2 {

    public:
    measure_G_tau_impr_est_v2(qmc_data const &data, int n_tau, gf_struct_t const &gf_struct,
                              many_body_op_t const &h_op,
                              container_set_t &results);
    void accumulate(mc_weight_t s);
    void collect_results(mpi::communicator const &c);

    private:
    qmc_data const &data;
    mc_weight_t average_sign;
    G_tau_G_target_t::view_type G_tau_with_O1_O2;
    G_tau_G_target_t::view_type G_tau_with_O1;
    G_tau_G_target_t::view_type G_tau_with_O2;
    op_desc h_op_d;

    // Pre-constructed commutator operators: maps (block_index, inner_index) -> op_desc
    std::map<std::pair<int, int>, op_desc> comm_H_c;      // [H_loc, c]
    std::map<std::pair<int, int>, op_desc> comm_H_cdag;   // [H_loc, c†]
  };

} // namespace triqs_cthyb
