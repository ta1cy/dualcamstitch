#pragma once
#include "orb_structures.h"
#include "cuda_utils.h"

namespace orb
{
    void setMaxNumPoints(const int num);
    void getPointCounter(void** addr);
    void setFastThresholdLUT(int fast_threshold);
    void setUmax(const int patch_size);
    void setPattern(const int patch_size, const int wta_k);
    void setGaussianKernel();
    void setScaleSqSq();
    void makeOffsets(int* pitchs, int noctaves);

    void hFastDectectWithNMS(unsigned char* image, unsigned char* octave_images, float* vmem, 
                              OrbData& result, int* oszp, int noctaves, int threshold, 
                              int border, bool harris_score);
    void hComputeAngle(unsigned char* octave_images, OrbData& result, int* oszp, 
                       int noctaves, int patch_size);
    void hGassianBlur(unsigned char* octave_images, int* oszp, int noctaves);
    void hDescribe(unsigned char* octave_images, OrbData& result, unsigned char* desc, 
                   int wta_k, int noctaves);
    void hMatch(OrbData& result1, OrbData& result2, unsigned char* desc1, unsigned char* desc2);
}
