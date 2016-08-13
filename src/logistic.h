
#pragma once
#include "image.h"
#include <algorithm>
#include <cmath>
#include <memory>
#include <cstring>
#include <random>
#include <array>

#include <cassert>

namespace logistic {

    static float clamp_f(float min, float max, float x)
    {
        return std::max(min, std::min(max, x));
    }

    const int C = 1;
    class LogisticRegression {
    public:
        LogisticRegression(int numClasses, int numFeatures, int w, int h)
            : numClasses(numClasses), numFeatures(numFeatures), gen(std::random_device()()), w(w), h(h), local_img(w,h)
        {
            weights = std::vector<float>(numClasses*numFeatures);
            auto dist = std::normal_distribution<float>(0, 1);
            for (auto & w : weights)
                w = dist(gen);
        }
        template<typename T>
        void computeMeans(const std::vector<img::Image<T, C>> & imgs)
        {
            scales = std::vector<float>(w*h, 0);
            biases = std::vector<float>(w*h, 0);
            int n = imgs.size();

            //compute means
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < w*h; j++) {
                    biases[j] += imgs[i].ptr[j];
                }
            }
            for (int j = 0; j < w*h; j++) {
                biases[j] /= n;
            }

            //compute variance
            auto sqr = [](float x){return x*x; };
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < w*h; j++) {
                    scales[j] += sqr(imgs[i].ptr[j]-biases[j]);
                }
            }
            for (int j = 0; j < w*h; j++) {
                scales[j] = scales[j] ? sqrt(scales[j]/n) : 1;
            }

        }
        std::vector<float> run(img::Image<float, C> & input) {
            std::vector<float> scores(numClasses,0);
            for (int i = 0; i < w*h; i++) {
                for (int c = 0; c < numClasses; c++) {
                    scores[c] += weights[c*numFeatures + i] * input.ptr[i];
                }
            }
            return scores;
        }
        template <typename T> int sgn(T val) {
            return (T(0) < val) - (val < T(0));
        }
        template<typename T>
        void train(std::vector<img::Image<T, C>> & imgs, std::vector<uint8_t> labels, int batch_num, float learning_rate, float l1_reg, float l2_reg) {
            std::uniform_int_distribution<int> dist(0, imgs.size() - 1);
            auto loss = 0.0f;
            std::vector<float> grad(numClasses*numFeatures,0);
            img::Image<float, 10> weightImg(w, h, weights.data());
            img::Image<float, 10> gradImg(w, h, grad.data());
            img::Image<float, 1> scalemg(w, h, scales.data());
            img::Image<float, 1> biasImg(w, h, biases.data());

            for (int i = 0; i < batch_num; i++) {
                auto idx = dist(gen);
                auto lbl = labels[idx];
                auto img = imgs[idx];

                auto scores = predict(img);
                loss += -std::log(scores[lbl]);


                // local image got updated in the predict call!!
                for (int j = 0; j < w*h; j++) {
                    for (int c = 0; c < numClasses; c++) {
                        grad[c*numFeatures + j] += (scores[c] - (lbl ==c)) * local_img.ptr[j];
                    }
                }
            }
            loss /= batch_num;
            for (auto & w : grad)
                w /= batch_num;

            float w1 = 0;
            float w2 = 0;
            for (auto & w : weights) {
                w1 += std::abs(w);
                w2 += w*w;
            }
            loss += l1_reg*w1 + 0.5f * l2_reg * w2;
            for (size_t i = 0; i < grad.size(); i++)
                grad[i] += l1_reg * sgn(weights[i]) + l2_reg * weights[i];

            // now apply
            for (size_t i = 0; i < weights.size(); i++)
                weights[i] -= learning_rate * grad[i];
        }
        template<typename T>
        std::vector<float> predict(const img::Image<T, C> & img) {
            for (int j = 0; j < w*h; j++) {
                local_img.ptr[j] = (img.ptr[j] - biases[j]) / scales[j];
            }
            auto scores = run(local_img);
            auto maxE = *std::max_element(scores.begin(), scores.end());
            auto sumE = 0.0f;
            for (auto & s : scores) {
                s = std::exp(s - maxE);
                sumE += s;
            }
            for (auto & s : scores)
                s = s / sumE;
            return scores;
        }

    private:
        img::Image<float, C> local_img;
        std::vector<float> weights; 
        std::vector<float> scales;
        std::vector<float> biases;

        std::mt19937 gen;
        int numFeatures, numClasses;
        int w, h;
    };
}