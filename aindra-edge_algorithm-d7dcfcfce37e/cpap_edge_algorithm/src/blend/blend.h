#ifndef LIST_TO_VEC_H
#define LIST_TO_VEC_H
#include <iostream>
#include <tuple>
#include <queue>

void create_blend_mask(int src_img_height, int src_img_width, void** alpha_mask);

int save_blank_img(std::string target_path, float scale, int tile_height, int tile_width);
int blend_and_save(void* src_tiles_vec, std::string target_path, float scale, int tile_height, int tile_widht);
int alpha_blend_and_save(void* src_tiles_vec, void* mask, std::string target_path, float scale, int tile_height, int tile_width);

#endif // LIST_TO_VEC_H
