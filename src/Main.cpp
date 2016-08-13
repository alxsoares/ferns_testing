#include "image_io.h"
#include "image_filter.h"
#include "fern.h"
#include "logistic.h"
#include "mnist.h"
#include <iostream>
int main(int argc, char * argv[])
{
	auto train_img = readMNISTImg("train-images.idx3-ubyte");
	auto train_lbl = readMNISTLabel("train-labels.idx1-ubyte");
	auto test_img = readMNISTImg("t10k-images.idx3-ubyte");
	auto test_lbl = readMNISTLabel("t10k-labels.idx1-ubyte");
	//auto test_img = readMNISTImg("t10k-images.idx3-ubyte");
	//auto test_lbl = readMNISTLabel("t10k-labels.idx1-ubyte");
	//Fern Features, Number of Ferns, Class Number
    ferns::FernClassifier fc(4, 20, 10, train_img[0].width, train_img[0].height);
    logistic::LogisticRegression lc(10, 28 * 28, train_img[0].width, train_img[0].height);

    printf("%d %d\n",train_img[0].width,train_img[0].height);
    fc.sampleFeatureFerns();

    ferns::FernClassifier bestFC = fc;
    float bestAcc = 1.0;
	float pAcc = 0;
	int iter = 0;
	int iterations = 15000;
	float startTmp = 1.0f;
	float minTmp = 0.00001f;
	float tmpFactor = 0.99f;
	float temp = startTmp;
	std::cout.precision(3);
    std::random_device rd;
    std::mt19937 gen(rd());

    float learning_rate = 0.1f;
    float l1_reg = 0.001f;
    float l2_reg = 0.001f;

    int batch_size = 64;

    lc.computeMeans(train_img);

	while (true) {
		fc = bestFC;

		//fc.sampleFeatureFerns(train_img[0].width, train_img[0].height);
		//fc.sampleOneFern(train_img[0].width, train_img[0].height);
		//fc.sampleOneFeature(train_img[0].width, train_img[0].height);
		for (size_t i = 0; i < train_img.size(); i++) {
			fc.train(train_img[i], train_lbl[i]);
		}
        fc.sampleBadFeatures();
        for (size_t i = 0; i < train_img.size()/4; i += batch_size)
            lc.train(train_img, train_lbl, batch_size, learning_rate, l1_reg, l2_reg);

		for (size_t i = 0; i < train_img.size(); i++) {
			fc.train(train_img[i], train_lbl[i]);
		}

		int predictQuality[10 * 10] = {};
		float correct = 0;
		for (size_t i = 0; i < test_img.size(); i++) {
			auto probs = fc.predict(test_img[i]);
            auto res = std::max_element(probs.begin(), probs.end()) - probs.begin();
			predictQuality[10 * res + test_lbl[i]]++;
			if (test_lbl[i] == res)
				correct++;
		}
		float meanAccuracy = correct / test_img.size();
		//std::cout << meanAccuracy << std::endl;
		auto sqr = [](float x) { return x*x; };
		auto CONST = 0.1f;
		//temp = CONST*sqr(((float)(iterations - iter)) / ((float)iterations));
		float cost = exp(-(pAcc - meanAccuracy) / temp);
		std::uniform_real_distribution<float> cDist(0, 1.0f);
		temp = std::max(tmpFactor*temp, minTmp);
		float sample = cDist(gen);

		//if (meanAccuracy > bestAcc)
		//{
		//	bestFC = fc;
		//	bestAcc = meanAccuracy;
		//	pAcc = meanAccuracy;
		//	std::cout << "NewBest" << std::endl;
		//}
		//else if (meanAccuracy > pAcc) {
		//	bestFC = fc;
		//	pAcc = meanAccuracy;
		//	std::cout << "acceptedBetter" << std::endl;
		//}
		//else if (cost > sample) {
		//	bestFC = fc;
		//	pAcc = meanAccuracy;
		//	std::cout << "accepted" << std::endl;
		//}
		//if (meanAccuracy > pAcc) 
		{
			bestFC = fc;
			pAcc = meanAccuracy;
			//std::cout << "accepted" << std::endl;
		}
        float correct_lc = 0;
        for (size_t i = 0; i < test_img.size(); i++) {
            auto probs = lc.predict(test_img[i]);
            auto res = std::max_element(probs.begin(), probs.end()) - probs.begin();
            if (test_lbl[i] == res)
                correct_lc++;
        }
        float meanAccuracy2 = correct_lc / test_img.size();

        std::cout << meanAccuracy << "\t" << meanAccuracy2 << "\t" << cost << "\t" << sample << "\t" << iter << "\t" << temp << std::endl;

		iter++;
		if (iter >= iterations) {
			iter = 0;
			fc.sampleFeatureFerns();
			bestFC = fc;
			pAcc = 0.0;
			temp = startTmp;
		}
	}

	return 0;
}
