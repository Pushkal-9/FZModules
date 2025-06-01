#ifndef D4543D36_B236_4E84_B7C9_AADFE5E84628
#define D4543D36_B236_4E84_B7C9_AADFE5E84628

#include <stddef.h>
#include <stdint.h>

#define FZGHEADER_HEADER 0
#define FZGHEADER_BITFLAG 1
#define FZGHEADER_START_POS 2
#define FZGHEADER_BITSTREAM 3
#define FZGHEADER_END 4

#define TODO_CHANGE_FZGHEADER_SIZE 128

typedef struct fzg_header {
  // Just use a single struct with the layout you need
  size_t original_len;
  uint32_t entry[FZGHEADER_END + 1];

  // Padding to ensure the struct is exactly 128 bytes
  uint8_t padding[TODO_CHANGE_FZGHEADER_SIZE - sizeof(size_t) -
                  sizeof(uint32_t) * (FZGHEADER_END + 1)];

  // Function to get raw bytes pointer when needed
  inline uint8_t* get_raw_bytes() { return reinterpret_cast<uint8_t*>(this); }
} fzg_header;

// Static assertion to verify the size
static_assert(sizeof(fzg_header) == TODO_CHANGE_FZGHEADER_SIZE,
              "fzg_header must be exactly 128 bytes");

#endif /* D4543D36_B236_4E84_B7C9_AADFE5E84628 */
