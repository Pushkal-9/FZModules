/**
 * @file hfclass.cu
 * @author Jiannan Tian
 * @brief
 * @version 0.3
 * @date 2023-06-02
 * (created) 2020-04-24
 *
 * @copyright (C) 2020 by Washington State University, The University of
 * Alabama, Argonne National Laboratory
 * @copyright (C) 2021 by Washington State University, Argonne National
 * Laboratory
 * @copyright (C) 2023 by Indiana University
 *
 */

#include "hfclass.hh"

// deps
#include <cuda.h>
// definitions
#include "hfclass.cuhip.inl"

template class phf::HuffmanCodec<uint8_t>;
template class phf::HuffmanCodec<uint16_t>;
template class phf::HuffmanCodec<uint32_t>;

