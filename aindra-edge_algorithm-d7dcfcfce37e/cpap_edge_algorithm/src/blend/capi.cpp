#include "spdlog/spdlog.h"
#include "spdlog/sinks/basic_file_sink.h"
#include "spdlog/async.h"

#include "capi.h"
#include "blend.h"

extern "C"
{

int initialize_logger(char* filename)
{
    auto file_logger = spdlog::basic_logger_mt<spdlog::async_factory>("blend_logger", std::string(filename));

    file_logger->set_level(spdlog::level::debug);
    file_logger->flush_on(spdlog::level::info);

    spdlog::get("blend_logger")->debug("Checking debug logging");
    spdlog::get("blend_logger")->info("Checking info logging");
    spdlog::get("blend_logger")->error("Checking error logging");
    return  0;
}

int allocate(void **data_vec)
{
    if (!(*data_vec))
        *data_vec = (void*)(new std::vector<std::tuple<std::string , int, int, int, int, int , int, int, int>>);

    return 0;
}

int deallocate(void *data_vec)
{
    std::vector<std::tuple<std::string , int,int,int,int,int ,int,int,int >>* ptr_data_vec;
    ptr_data_vec = (std::vector<std::tuple<std::string , int,int,int,int,int ,int,int,int >>*)(data_vec);
    ptr_data_vec->clear();
    delete ptr_data_vec;
    return 0;
}

int append( void** data_vec, char* src_img_path,
                  int img_tl_rel_0,int img_tl_rel_1,
                  int img_br_rel_0, int img_br_rel_1,
                  int tile_tl_rel_0, int tile_tl_rel_1,
                  int tile_br_rel_0, int tile_br_rel_1)
{
    try {
        static_cast<std::vector<std::tuple<std::string, int, int, int, int, int, int, int, int>>*>(*data_vec)->push_back(std::make_tuple(src_img_path,
                                                                                                                                         img_tl_rel_0, img_tl_rel_1,
                                                                                                                                         img_br_rel_0, img_br_rel_1,
                                                                                                                                         tile_tl_rel_0, tile_tl_rel_1,
                                                                                                                                         tile_br_rel_0, tile_br_rel_1));
        return 0;
    } catch (...) {
        return 1;
    }

    return 0;
}

int create_alpha_mask(int src_img_height, int src_img_width, void** mask_ptr)
{
    try {
        spdlog::get("blend_logger")->debug("Creating mask for linear blending");
        create_blend_mask(src_img_height, src_img_width, mask_ptr);
        spdlog::get("blend_logger")->info("Mask created for linear blending");
        return 0;
    } catch (...) {
        spdlog::get("blend_logger")->error("Mask creation for linear blending failed");
        return 1;
    }

}

int save_blank_image(char* target_path, float scale, int height, int width)
{
    try {
        int status = save_blank_img(std::string(target_path), scale, height, width);
        if(status == 1)
        {
            return 0;
        }
    } catch (...) {
        return 1;
    }

}

int save_avg_blended_image(void* src_tiles_vec, char* target_path, float scale, int height, int width)
{
    try {
        int status = blend_and_save(src_tiles_vec, std::string(target_path), scale, height, width);

        if(status == 1)
        {
            return 0;
        }
    } catch (...) {
        return 1;
    }

}

int save_alpha_blended_image(void* src_tiles_vec, void* mask, char* target_path, float scale, int tile_height, int tile_width)
{
    try {
        int status = alpha_blend_and_save(src_tiles_vec, mask, std::string(target_path), scale, tile_height, tile_width);


        if(status == 1)
        {
            return 0;
        }
    } catch (...) {
        return 1;
    }
}

int remove_logger(){
    spdlog::drop_all();
}
}
