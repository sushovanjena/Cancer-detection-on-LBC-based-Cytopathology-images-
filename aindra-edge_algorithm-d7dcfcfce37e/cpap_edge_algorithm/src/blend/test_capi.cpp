#include <iostream>
# include "capi.h"

int main()
{
    initialize_logger((char*)"/home/aindra/abhay/stitch_algo_log.log_30-12-2019_16-15-38.clog");
    void *mask = 0;
    create_alpha_mask(2048, 2448, &mask);
    void *src_vec = 0;
    allocate(&src_vec);
    append(&src_vec, (char*)"/home/aindra/abhay/000397_49_6_395.jpg", 0,0,779,961,182,0,961,961);
    append(&src_vec, (char*)"/home/aindra/abhay/000397_49_6_395.jpg", 0,0,779,961,182,0,961,961);
    append(&src_vec, (char*)"/home/aindra/abhay/000397_49_6_395.jpg", 778,0,1740,961,0,0,962,961);

    save_alpha_blended_image(src_vec, mask, (char*)"/home/aindra/abhay/0_2.jpg", 1.0, 962, 961);
    deallocate(src_vec);
    remove_logger();

    return 0;
}
