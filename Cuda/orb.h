#pragma once
#include "orb_structures.h"
#include "cuda_utils.h"
#include <vector>

namespace orb
{
    class Orbor
    {
    public:
        Orbor();
        ~Orbor();

        void init(int _noctaves = 5, int _edge_threshold = 31, int _wta_k = 2, 
                  ScoreType _score_type = ScoreType::HARRIS_SCORE,
                  int _patch_size = 31, int _fast_threshold = 20, 
                  int _retain_topn = -1, int _max_pts = 1000);

        void detectAndCompute(unsigned char* image, OrbData& result, int3 whp0, 
                              void** desc_addr = NULL, const bool compute_desc = true);

        void match(OrbData& result1, OrbData& result2, unsigned char* desc1, unsigned char* desc2);

        void initOrbData(OrbData& data, const int max_pts, const bool host, const bool dev);
        void freeOrbData(OrbData& data);

        float last_detect_ms = 0;
        float last_describe_ms = 0;
        float last_match_ms = 0;

    private:
        int noctaves = 5;
        int max_octave = 5;
        int edge_threshold = 31;
        int wta_k = 2;
        ScoreType score_type = ScoreType::HARRIS_SCORE;
        int patch_size = 31;
        int fast_threshold = 20;
        int retain_topn = -1;
        int max_pts = 1000;
        int width = -1;
        int height = -1;
        unsigned char* omem = NULL;
        float* vmem = NULL;
        size_t obytes = 0;
        size_t vbytes = 0;
        std::vector<int> oszp;
        unsigned int* d_point_counter_addr = NULL;

        void updateParam(int3 whp0);
        void detect(unsigned char* image, OrbData& result);
    };
}
