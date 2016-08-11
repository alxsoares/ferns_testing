#pragma once
#include "image.h"
#ifdef _WIN32 
#include <intrin.h>
#define bSwap _byteswap_ulong
#else
#define bSwap __builtin_bswap32
#endif
#include <vector>
#include <fstream>
#include <random>
#include <tuple>
#include <algorithm>

std::vector<img::Img<uint8_t>> readMNISTImg(const std::string &fileName)
{
    std::vector<img::Img<uint8_t>> imgs;
    std::ifstream file(fileName, std::ios::binary);
    if (file.is_open())
    {
        uint32_t magic_number = 0;
        uint32_t number_of_images = 0;
        uint32_t n_rows = 0;
        uint32_t n_cols = 0;
        file.read((char*)&magic_number, sizeof(magic_number));
        magic_number = bSwap(magic_number);
        file.read((char*)&number_of_images, sizeof(number_of_images));
        number_of_images = bSwap(number_of_images);
        file.read((char*)&n_rows, sizeof(n_rows));
        n_rows = bSwap(n_rows);
        file.read((char*)&n_cols, sizeof(n_cols));
        n_cols = bSwap(n_cols);
        for (uint32_t i = 0; i<number_of_images; ++i)
        {
            img::Img<uint8_t> img(n_cols, n_rows);
            file.read((char*)img.data.get(), n_rows*n_cols);
            imgs.push_back(img);
        }
    }

    return imgs;
}

std::vector<uint8_t> readMNISTLabel(const std::string &fileName)
{
    std::vector<uint8_t> labels;
    std::ifstream file(fileName, std::ios::binary);
    if (file.is_open())
    {
        uint32_t magic_number = 0;
        uint32_t n_items = 0;
        file.read((char*)&magic_number, sizeof(magic_number));
        magic_number = bSwap(magic_number);
        file.read((char*)&n_items, sizeof(n_items));
        n_items = bSwap(n_items);
        labels.resize(n_items);
        file.read((char*)labels.data(), n_items);
    }
    return labels;
}