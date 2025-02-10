/**
 * @file dbg.hh
 * @author Jiannan Tian
 * @brief
 * @version 0.9
 * @date 2023-09-07
 *
 * (C) 2023 by Indiana University, Argonne National Laboratory
 *
 */

#ifndef E37962D3_F63F_4392_B670_41E4B31915ED
#define E37962D3_F63F_4392_B670_41E4B31915ED

#define __PSZLOG__P(STR) cout << STR << endl;

#define __PSZLOG__NEWLINE printf("\n");

// clang-format off
#define __PSZLOG__STATUS_INFO              printf("[fzmod::info] ");
#define __PSZLOG__STATUS_INFO_IN(WHAT)     printf("[fzmod::info::%s] ", WHAT);
#define __PSZLOG__STATUS_DBG               printf("[fzmod::\e[31mdbg\e[0m] ");
#define __PSZLOG__STATUS_DBG_IN(WHAT)      printf("[fzmod::\e[31mdbg\e[0m::%s] ", WHAT);
#define __PSZLOG__STATUS_SANITIZE          printf("[fzmod::\e[31mdbg::sanitize\e[0m] ");
#define __PSZLOG__STATUS_SANITIZE_IN(WHAT) printf("[fzmod::\e[31mdbg::sanitize::%s\e[0m] ", WHAT);
// clang-format on

#define __PSZLOG__WHERE_CALLED \
  printf("(func) %s at \e[31m%s:%d\e[0m\n", __func__, __FILE__, __LINE__);

#define __PSZDBG__INFO(STR) \
  {                         \
    __PSZLOG__NEWLINE;      \
    __PSZLOG__STATUS_DBG;   \
    __PSZLOG__WHERE_CALLED; \
    __PSZLOG__STATUS_DBG;   \
    cout << STR << endl;    \
  }

#define __PSZDBG__FATAL(CONST_CHAR) \
  throw std::runtime_error("\e[31m[fzmod::fatal]\e[0m " + string(CONST_CHAR));

#define __PSZDBG_VAR(WHAT, VAR)               \
  __PSZLOG__STATUS_DBG_IN(WHAT)               \
  printf("(var) " #VAR "=%ld", (int64_t)VAR); \
  __PSZLOG__NEWLINE;

#define __PSZDBG_PTR_WHERE(VAR)                                   \
  {                                                               \
    cudaPointerAttributes attributes;                             \
    cudaError_t err = cudaPointerGetAttributes(&attributes, VAR); \
    if (err != cudaSuccess) {                                     \
      __PSZLOG__STATUS_DBG                                        \
      cerr << "failed checking pointer attributes: "              \
           << cudaGetErrorString(err) << endl;                    \
    } else {                                                      \
      __PSZLOG__STATUS_DBG                                        \
      if (attributes.type == cudaMemoryTypeDevice)                \
        printf("(var) " #VAR " is on CUDA Device.");              \
      else if (attributes.type == cudaMemoryTypeDevice)           \
        printf("(var) " #VAR " is on Host.");                     \
      else {                                                      \
        printf("(var) " #VAR " is in another universe.");         \
      }                                                           \
      __PSZLOG__NEWLINE                                           \
    }                                                             \
  }

#ifdef PSZ_DBG_ON

#define PSZDBG_LOG(...) __PSZDBG__INFO(__VA_ARGS__)

#define PSZDBG_VAR(...)   \
  __PSZLOG__NEWLINE;      \
  __PSZLOG__STATUS_DBG    \
  __PSZLOG__WHERE_CALLED; \
  __PSZDBG_VAR(__VA_ARGS__);

#define PSZDBG_PTR_WHERE(VAR) \
  __PSZLOG__NEWLINE;          \
  __PSZLOG__STATUS_DBG        \
  __PSZLOG__WHERE_CALLED;     \
  __PSZDBG_PTR_WHERE(VAR)

#else
#define PSZDBG_LOG(...)
#define PSZDBG_VAR(...)
#define PSZDBG_PTR_WHERE(...)
#endif

#endif /* E37962D3_F63F_4392_B670_41E4B31915ED */
