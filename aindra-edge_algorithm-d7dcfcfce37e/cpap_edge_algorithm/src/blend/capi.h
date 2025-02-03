#ifndef CAPI_H
#define CAPI_H

#include <stdint.h>
#include "blend.h"

extern "C"
{
    int initialize_logger(char* filename);
    int allocate(void **data_vec);
    int append(void** data_vec, char* src_img_path,
                      int img_tl_rel_0,int img_tl_rel_1,
                      int img_br_rel_0, int img_br_rel_1,
                      int tile_tl_rel_0, int tile_tl_rel_1,
                      int tile_br_rel_0, int tile_br_rel_1);
    int create_alpha_mask(int height, int width, void** mask_ptr);
    int save_blank_image(char* target_path, float scale, int height, int width);
    int save_avg_blended_image(void* src_tiles_vec, char* target_path, float scale, int height, int width);
    int save_alpha_blended_image(void* src_tiles_vec, void* mask, char* target_path, float scale, int height, int width);
    int deallocate(void* data_vec);
    int remove_logger();
}

#endif // CAPI_H
