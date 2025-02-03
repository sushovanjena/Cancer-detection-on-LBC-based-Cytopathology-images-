#include <iostream>
#include <tuple>
#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>
#include "opencv2/highgui/highgui.hpp"
#include <opencv2/imgproc/imgproc.hpp>


#include "blend.h"

void create_blend_mask(int height, int width, void** alpha_mask)
{
    *alpha_mask = (void*)(new cv::Mat);

    int n = 128;
    cv::Mat w_mat = cv::Mat::ones(cv::Size(n, n), CV_8UC1);

    for(int i=1; i<n/2; i++)
    {
        cv::Rect roi_tile(i, i, n-2*i, n-2*i);
        w_mat(roi_tile) = i+1;
    }

    cv::resize(w_mat, w_mat, cv::Size(width, height), cv::INTER_NEAREST);
    w_mat.convertTo(w_mat, CV_16UC1);

    std::vector<cv::Mat> channels;
    channels.push_back(w_mat);
    channels.push_back(w_mat);
    channels.push_back(w_mat);

    cv::Mat mask;
    cv::merge(channels, mask);

    static_cast<cv::Mat*>(*alpha_mask)->push_back(mask);
}

int save_blank_img(std::string target_path, float scale, int height, int width)
{
    cv::Mat tile_template = cv::Mat(cv::Size(width, height), CV_8UC3, cv::Scalar(255,255, 255));

    if(scale != 1.0f)
    {
        height = int(height * scale);
        width = int(width * scale);
        cv::resize(tile_template, tile_template, cv::Size(width, height));
    }

    std::vector<int> compression_params {cv::IMWRITE_WEBP_QUALITY, 80};;
    int status = cv::imwrite(target_path, tile_template, compression_params);
    return status;
}

// int blend_and_save(void* src_tiles_vec, std::string target_path, float scale, int height, int width)
// {
//     cv::Mat tile_template = cv::Mat::zeros(cv::Size(width, height), CV_16UC3);
//     cv::Mat tile_template_mask = cv::Mat::zeros(cv::Size(width, height), CV_8UC1);

//     std::vector<std::tuple<std::string, int, int, int, int, int, int, int, int >>* src_vec = (std::vector<std::tuple<std::string, int, int, int, int, int, int, int, int >>*)src_tiles_vec;

//     for(int i=0; i < src_vec->size(); i++)
//     {
//         auto src_tile_tuple = src_vec->at(i);

//         std::string src_img_path = std::get<0>(src_tile_tuple);
//         int img_tl_rel_y = std::get<1>(src_tile_tuple);
//         int img_tl_rel_x = std::get<2>(src_tile_tuple);
//         int img_br_rel_y = std::get<3>(src_tile_tuple);
//         int img_br_rel_x = std::get<4>(src_tile_tuple);
//         int tile_tl_rel_y = std::get<5>(src_tile_tuple);
//         int tile_tl_rel_x = std::get<6>(src_tile_tuple);
//         int tile_br_rel_y = std::get<7>(src_tile_tuple);
//         int tile_br_rel_x = std::get<8>(src_tile_tuple);

//         cv::Mat img = cv::imread(src_img_path);

//         cv::Rect roi_tile(tile_tl_rel_x, tile_tl_rel_y, tile_br_rel_x-tile_tl_rel_x, tile_br_rel_y-tile_tl_rel_y);

//         cv::Rect roi_img(img_tl_rel_x, img_tl_rel_y, img_br_rel_x-img_tl_rel_x, img_br_rel_y-img_tl_rel_y);

//         cv::Mat img_roi_mat;
//         img_roi_mat = img(roi_img);

//         if(!img_roi_mat.empty())
//         {
//             tile_template(roi_tile) +=  img_roi_mat;
//             tile_template_mask(roi_tile) += 1;
//         }
//     }

//     cv::Mat zero_val_mask = tile_template_mask == 0;

//     tile_template_mask.convertTo(tile_template_mask, CV_16UC1);
//     std::vector<cv::Mat> channels;

//     channels.push_back(tile_template_mask);
//     channels.push_back(tile_template_mask);
//     channels.push_back(tile_template_mask);

//     cv::Mat tile_mask;
//     cv::merge(channels, tile_mask);

//     cv::divide(tile_template, tile_mask, tile_template);

//     tile_template.setTo(cv::Scalar(255, 255, 255), zero_val_mask);

//     if(scale != 1.0f)
//     {
//         height = int(height * scale);
//         width = int(width * scale);
//         cv::resize(tile_template, tile_template, cv::Size(width, height));
//     }

//     std::vector<int> compression_params {cv::IMWRITE_JPEG_QUALITY, 80};;
//     int status = cv::imwrite(target_path, tile_template, compression_params);
//     return status;
// }

int blend_and_save(void* src_tiles_vec, std::string target_path, float scale, int height, int width)
{
    cv::setNumThreads(0);
    // Print the function parameters for debugging
    std::cout << "Target path: " << target_path << std::endl;
    std::cout << "Scale: " << scale << std::endl;
    std::cout << "Height: " << height << ", Width: " << width << std::endl;

    cv::Mat tile_template = cv::Mat::zeros(cv::Size(width, height), CV_16UC3);
    cv::Mat tile_template_mask = cv::Mat::zeros(cv::Size(width, height), CV_8UC1);

    std::vector<std::tuple<std::string, int, int, int, int, int, int, int, int >>* src_vec = 
        (std::vector<std::tuple<std::string, int, int, int, int, int, int, int, int >>*)src_tiles_vec;

    std::cout << "Number of tiles: " << src_vec->size() << std::endl;

    for(int i=0; i < src_vec->size(); i++)
    {
        auto src_tile_tuple = src_vec->at(i);

        std::string src_img_path = std::get<0>(src_tile_tuple);
        int img_tl_rel_y = std::get<1>(src_tile_tuple);
        int img_tl_rel_x = std::get<2>(src_tile_tuple);
        int img_br_rel_y = std::get<3>(src_tile_tuple);
        int img_br_rel_x = std::get<4>(src_tile_tuple);
        int tile_tl_rel_y = std::get<5>(src_tile_tuple);
        int tile_tl_rel_x = std::get<6>(src_tile_tuple);
        int tile_br_rel_y = std::get<7>(src_tile_tuple);
        int tile_br_rel_x = std::get<8>(src_tile_tuple);

        // Print the source image path and ROIs for debugging
        std::cout << "Processing tile " << i+1 << " with image path: " << src_img_path << std::endl;
        std::cout << "Image ROI: (" << img_tl_rel_x << ", " << img_tl_rel_y << ") to (" << img_br_rel_x << ", " << img_br_rel_y << ")" << std::endl;
        std::cout << "Tile ROI: (" << tile_tl_rel_x << ", " << tile_tl_rel_y << ") to (" << tile_br_rel_x << ", " << tile_br_rel_y << ")" << std::endl;

        cv::Mat img = cv::imread(src_img_path);

        if (img.empty())
        {
            std::cerr << "Error loading image: " << src_img_path << std::endl;
            continue;  // Skip to the next tile if the image couldn't be loaded
        }

        cv::Rect roi_tile(tile_tl_rel_x, tile_tl_rel_y, tile_br_rel_x-tile_tl_rel_x, tile_br_rel_y-tile_tl_rel_y);
        cv::Rect roi_img(img_tl_rel_x, img_tl_rel_y, img_br_rel_x-img_tl_rel_x, img_br_rel_y-img_tl_rel_y);

        // Debug ROI sizes
        std::cout << "ROI tile size: " << roi_tile.width << "x" << roi_tile.height << std::endl;
        std::cout << "ROI image size: " << roi_img.width << "x" << roi_img.height << std::endl;

        cv::Mat img_roi_mat;
        img_roi_mat = img(roi_img);

        if(!img_roi_mat.empty())
        {
            // Print matrix size for debugging
            std::cout << "img_roi_mat size: " << img_roi_mat.size() << std::endl;
            
            tile_template(roi_tile) += img_roi_mat;
            tile_template_mask(roi_tile) += 1;
        }
        else
        {
            std::cerr << "img_roi_mat is empty for tile " << i+1 << std::endl;
        }
    }

    cv::Mat zero_val_mask = tile_template_mask == 0;

    tile_template_mask.convertTo(tile_template_mask, CV_16UC1);
    std::vector<cv::Mat> channels;

    channels.push_back(tile_template_mask);
    channels.push_back(tile_template_mask);
    channels.push_back(tile_template_mask);

    cv::Mat tile_mask;
    cv::merge(channels, tile_mask);

    std::cout << "Dividing tile_template by tile_mask" << std::endl;
    cv::divide(tile_template, tile_mask, tile_template);


    tile_template.setTo(cv::Scalar(255, 255, 255), zero_val_mask);
    try
    {
        if (scale != 1.0f)
        {
            std::cout << "Before resizing: " << tile_template.cols << "x" << tile_template.rows << std::endl;
            std::cout << "tile_template type: " << tile_template.type() << std::endl;
            std::cout << "tile_template channels: " << tile_template.channels() << std::endl;
            height = int(height * scale);
            width = int(width * scale);
            std::cout << "Resizing to new dimensions: " << width << "x" << height << std::endl;
            cv::resize(tile_template, tile_template, cv::Size(width, height));
            std::cout << "After resizing: " << tile_template.cols << "x" << tile_template.rows << std::endl;
            std::cout << "Resized to new dimensions." << std::endl;
        }
    }
    catch (const cv::Exception& e)
    {
        std::cerr << "OpenCV error: " << e.what() << std::endl;
    }
    catch (const std::exception& e)
    {
        std::cerr << "Standard exception: " << e.what() << std::endl;
    }
    catch (...)
    {
        std::cerr << "Unknown error occurred." << std::endl;
    }

    std::cout << "Compressing to JPEG quality." << std::endl;
    std::vector<int> compression_params {cv::IMWRITE_WEBP_QUALITY, 80};

    std::cout << "Saving the blended image to: " << target_path << std::endl;
    int status = cv::imwrite(target_path, tile_template, compression_params);

    if (status)
        std::cout << "Image saved successfully." << std::endl;
    else
        std::cerr << "Failed to save image." << std::endl;

    return status;
}

int alpha_blend_and_save(void* src_tiles_vec, void* mask, std::string target_path, float scale, int height, int width)
{

    cv::Mat mask_1_channel;
    cv::Mat mask_3_channel;

    cv::Mat tile_template = cv::Mat::zeros(cv::Size(width, height), CV_16UC3);
    cv::Mat tile_template_mask = cv::Mat::zeros(cv::Size(width, height), CV_16UC1);

    cv::Mat alpha_mask;
    ((cv::Mat*)(mask))->assignTo(alpha_mask);

    std::vector<cv::Mat> channels(3);
    cv::split(alpha_mask, channels);

    mask_1_channel = channels[0];
    mask_3_channel = alpha_mask;

    std::vector<std::tuple<std::string, int, int, int, int, int, int, int, int >>* src_vec = (std::vector<std::tuple<std::string, int, int, int, int, int, int, int, int >>*)src_tiles_vec;

    for(int i=0; i < src_vec->size(); i++)
    {
        auto src_tile_tuple = src_vec->at(i);

        std::string src_img_path = std::get<0>(src_tile_tuple);
        int img_tl_rel_y = std::get<1>(src_tile_tuple);
        int img_tl_rel_x = std::get<2>(src_tile_tuple);
        int img_br_rel_y = std::get<3>(src_tile_tuple);
        int img_br_rel_x = std::get<4>(src_tile_tuple);
        int tile_tl_rel_y = std::get<5>(src_tile_tuple);
        int tile_tl_rel_x = std::get<6>(src_tile_tuple);
        int tile_br_rel_y = std::get<7>(src_tile_tuple);
        int tile_br_rel_x = std::get<8>(src_tile_tuple);

        cv::Mat img = cv::imread(src_img_path);

        cv::Rect roi_tile(tile_tl_rel_x, tile_tl_rel_y, tile_br_rel_x-tile_tl_rel_x, tile_br_rel_y-tile_tl_rel_y);

        cv::Rect roi_img(img_tl_rel_x, img_tl_rel_y, img_br_rel_x-img_tl_rel_x, img_br_rel_y-img_tl_rel_y);

        cv::Mat img_roi_mat = img(roi_img);

        // Region calcualtion for contribution from each image is coming 0 sometime
        if(!img_roi_mat.empty())
        {
            img_roi_mat.convertTo(img_roi_mat, CV_16UC3);
            tile_template(roi_tile) +=  img_roi_mat.mul(mask_3_channel(roi_img));
            tile_template_mask(roi_tile) += mask_1_channel(roi_img);
        }
    }


    cv::Mat zero_val_mask = tile_template_mask == 0;

    std::vector<cv::Mat> mask_template;
    mask_template.push_back(tile_template_mask);
    mask_template.push_back(tile_template_mask);
    mask_template.push_back(tile_template_mask);

    cv::Mat tile_mask;
    cv::merge(mask_template, tile_mask);
    cv::divide(tile_template, tile_mask, tile_template);
    tile_template.setTo(cv::Scalar(255, 255, 255), zero_val_mask);

    if(scale != 1.0f)
    {
        height = int(height * scale);
        width = int(width * scale);
        cv::resize(tile_template, tile_template, cv::Size(width, height));
    }    

    std::vector<int> compression_params {cv::IMWRITE_WEBP_QUALITY, 80};
    int status = cv::imwrite(target_path, tile_template, compression_params);
    return  status;
}
