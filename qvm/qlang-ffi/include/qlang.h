#ifndef QLANG_H
#define QLANG_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

int32_t qlang_create(size_t num_qubits);
int32_t qlang_run_source(const char *source);
int32_t qlang_reset(void);
size_t qlang_num_qubits(void);
ptrdiff_t qlang_measure_all(uint8_t *output, size_t capacity);
char *qlang_state_json(void);
const char *qlang_last_error(void);
void qlang_string_free(char *value);

#ifdef __cplusplus
}
#endif

#endif
