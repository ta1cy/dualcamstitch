#pragma once

namespace orb
{
    struct OrbPoint
    {
        int x;
        int y;
        int octave;
        float score;
        float angle;
        int match;
        int distance;
    };

    struct OrbData
    {
        int num_pts;
        OrbPoint* h_data;
        OrbPoint* d_data;
    };

    enum ScoreType
    {
        HARRIS_SCORE = 0,
        FAST_SCORE = 1
    };
}
