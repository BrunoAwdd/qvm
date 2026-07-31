#ifndef QLANG_H
#define QLANG_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct QLangHandle QLangHandle;

QLangHandle *qlang_create(size_t num_qubits);
void qlang_destroy(QLangHandle *handle);
int32_t qlang_run_source(QLangHandle *handle, const char *source);
int32_t qlang_reset(QLangHandle *handle);
size_t qlang_num_qubits(QLangHandle *handle);
ptrdiff_t qlang_measure_all(QLangHandle *handle, uint8_t *output, size_t capacity);
char *qlang_state_json(QLangHandle *handle);
const char *qlang_last_error(void);
void qlang_string_free(char *value);

#ifdef __cplusplus
}
#endif

#endif
