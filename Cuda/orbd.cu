#include "orbd.h"
#include <device_launch_parameters.h>

namespace orb
{

#define X1          64
#define X2          32
#define MAX_OCTAVE  5
#define FAST_PATTERN 16
#define HARRIS_SIZE 7
#define MAX_PATCH   31
#define K           (FAST_PATTERN / 2)
#define N           (FAST_PATTERN + K + 1)
#define HARRIS_K    0.04f
#define MAX_DIST    64
#define GR          3
#define R2          6
#define R4          12
#define DX          (X2 - R2)

__constant__ int d_max_num_points;
__constant__ float d_scale_sq_sq;
__device__ unsigned int d_point_counter;
__constant__ int dpixel[25 * MAX_OCTAVE];
__constant__ unsigned char dthresh_table[512];
__constant__ int d_umax[MAX_PATCH / 2 + 2];
__constant__ int2 d_pattern[512];
__constant__ float d_gauss[GR + 1];
__constant__ int ofs[HARRIS_SIZE * HARRIS_SIZE];
__constant__ int angle_param[MAX_OCTAVE * 2];

void setMaxNumPoints(const int num)
{
    CHECK(cudaMemcpyToSymbol(d_max_num_points, &num, sizeof(int), 0, cudaMemcpyHostToDevice));
}

void getPointCounter(void** addr)
{
    CHECK(cudaGetSymbolAddress(addr, d_point_counter));
}

void setFastThresholdLUT(int fast_threshold)
{
    unsigned char hthreshold_tab[512];
    for (int i = -255, j = 0; i <= 255; i++, j++)
        hthreshold_tab[j] = (unsigned char)(i < -fast_threshold ? 1 : i > fast_threshold ? 2 : 0);
    CHECK(cudaMemcpyToSymbol(dthresh_table, hthreshold_tab, 512 * sizeof(unsigned char), 0, cudaMemcpyHostToDevice));
}

void setUmax(const int patch_size)
{
    int half_patch = patch_size / 2;
    int* h_umax = new int[half_patch + 2];
    h_umax[half_patch + 1] = 0;

    float v = half_patch * sqrtf(2.f) / 2;
    int vmax = (int)floorf(v + 1);
    int vmin = (int)ceilf(v);
    for (int i = 0; i <= vmax; i++)
        h_umax[i] = (int)roundf(sqrtf(half_patch * half_patch - i * i));

    for (int i = half_patch, v0 = 0; i >= vmin; --i)
    {
        while (h_umax[v0] == h_umax[v0 + 1])
            ++v0;
        h_umax[i] = v0;
        ++v0;
    }

    CHECK(cudaMemcpyToSymbol(d_umax, h_umax, sizeof(int) * (half_patch + 2), 0, cudaMemcpyHostToDevice));
    delete[] h_umax;
}

void setPattern(const int patch_size, const int wta_k)
{
    int bit_pattern_31_[256 * 4] = {
        8,-3, 9,5, 4,2, 7,-12, -11,9, -8,2, 7,-12, 12,-13,
        2,-13, 2,12, 1,-7, 1,6, -2,-10, -2,-4, -13,-13, -11,-8,
        -13,-3, -12,-9, 10,4, 11,9, -13,-8, -8,-9, -11,7, -9,12,
        7,7, 12,6, -4,-5, -3,0, -13,2, -12,-3, -9,0, -7,5,
        12,-6, 12,-1, -3,6, -2,12, -6,-13, -4,-8, 11,-13, 12,-8,
        4,7, 5,1, 5,-3, 10,-3, 3,-7, 6,12, -8,-7, -6,-2,
        -2,11, -1,-10, -13,12, -8,10, -7,3, -5,-3, -4,2, -3,7,
        -10,-12, -6,11, 5,-12, 6,-7, 5,-6, 7,-1, 1,0, 4,-5,
        9,11, 11,-13, 4,7, 4,12, 2,-1, 4,4, -4,-12, -2,7,
        -8,-5, -7,-10, 4,11, 9,12, 0,-8, 1,-13, -13,-2, -8,2,
        -3,-2, -2,3, -6,9, -4,-9, 8,12, 10,7, 0,9, 1,3,
        7,-5, 11,-10, -13,-6, -11,0, 10,7, 12,1, -6,-3, -6,12,
        10,-9, 12,-4, -13,8, -8,-12, -13,0, -8,-4, 3,3, 7,8,
        5,7, 10,-7, -1,7, 1,-12, 3,-10, 5,6, 2,-4, 3,-10,
        -13,0, -13,5, -13,-7, -12,12, -13,3, -11,8, -7,12, -4,7,
        6,-10, 12,8, -9,-1, -7,-6, -2,-5, 0,12, -12,5, -7,5,
        3,-10, 8,-13, -7,-7, -4,5, -3,-2, -1,-7, 2,9, 5,-11,
        -11,-13, -5,-13, -1,6, 0,-1, 5,-3, 5,2, -4,-13, -4,12,
        -9,-6, -9,6, -12,-10, -8,-4, 10,2, 12,-3, 7,12, 12,12,
        -7,-13, -6,5, -4,9, -3,4, 7,-1, 12,2, -7,6, -5,1,
        -13,11, -12,5, -3,7, -2,-6, 7,-8, 12,-7, -13,-7, -11,-12,
        1,-3, 12,12, 2,-6, 3,0, -4,3, -2,-13, -1,-13, 1,9,
        7,1, 8,-6, 1,-1, 3,12, 9,1, 12,6, -1,-9, -1,3,
        -13,-13, -10,5, 7,7, 10,12, 12,-5, 12,9, 6,3, 7,11,
        5,-13, 6,10, 2,-12, 2,3, 3,8, 4,-6, 2,6, 12,-13,
        9,-12, 10,3, -8,4, -7,9, -11,12, -4,-6, 1,12, 2,-8,
        6,-9, 7,-4, 2,3, 3,-2, 6,3, 11,0, 3,-3, 8,-8,
        7,8, 9,3, -11,-5, -6,-4, -10,11, -5,10, -5,-8, -3,12,
        -10,5, -9,0, 8,-1, 12,-6, 4,-6, 6,-11, -10,12, -8,7,
        4,-2, 6,7, -2,0, -2,12, -5,-8, -5,2, 7,-6, 10,12,
        -9,-13, -8,-8, -5,-13, -5,-2, 8,-8, 9,-13, -9,-11, -9,0,
        1,-8, 1,-2, 7,-4, 9,1, -2,1, -1,-4, 11,-6, 12,-11
    };

    const int npoints = 512;
    int2 patternbuf[npoints];
    const int2* pattern0 = (const int2*)bit_pattern_31_;
    
    if (patch_size != 31)
    {
        pattern0 = patternbuf;
        srand(0x34985739);
        for (int i = 0; i < npoints; i++)
        {
            patternbuf[i].x = rand() % patch_size - patch_size / 2;
            patternbuf[i].y = rand() % patch_size - patch_size / 2;
        }
    }

    if (wta_k == 2)
        CHECK(cudaMemcpyToSymbol(d_pattern, pattern0, npoints * sizeof(int2), 0, cudaMemcpyHostToDevice));
    else
    {
        srand(0x12345678);
        int ntuples = 32 * 4;
        int2* pattern = new int2[ntuples * wta_k];
        for (int i = 0; i < ntuples; i++)
        {
            for (int k = 0; k < wta_k; k++)
            {
                while (true)
                {
                    int idx = rand() % npoints;
                    int2 pt = pattern0[idx];
                    int k1;
                    for (k1 = 0; k1 < k; k1++)
                    {
                        int2 pt1 = pattern[wta_k * i + k1];
                        if (pt.x == pt1.x && pt.y == pt1.y)
                            break;
                    }
                    if (k1 == k)
                    {
                        pattern[wta_k * i + k] = pt;
                        break;
                    }
                }
            }
        }
        CHECK(cudaMemcpyToSymbol(d_pattern, pattern, ntuples * wta_k * sizeof(int2), 0, cudaMemcpyHostToDevice));
        delete[] pattern;
    }
}

void setGaussianKernel()
{
    const float sigma = 2.f;
    const float svar = -1.f / (2.f * sigma * sigma);
    float kernel[GR + 1];
    float kersum = 0.f;
    for (int i = 0; i <= GR; i++)
    {
        kernel[i] = expf(i * i * svar);
        kersum = kersum + (kernel[i] + (i == 0 ? 0 : kernel[i]));
    }
    kersum = 1.f / kersum;
    for (int i = 0; i <= GR; i++)
        kernel[i] *= kersum;
    CHECK(cudaMemcpyToSymbol(d_gauss, kernel, (GR + 1) * sizeof(float), 0, cudaMemcpyHostToDevice));
}

void setScaleSqSq()
{
    float scale = 1.f / (4 * HARRIS_SIZE * 255.f);
    float scale_sq_sq = scale * scale * scale * scale;
    CHECK(cudaMemcpyToSymbol(d_scale_sq_sq, &scale_sq_sq, sizeof(float), 0, cudaMemcpyHostToDevice));
}

void setHarrisOffsets(const int pitch)
{
    static int p = -1;
    if (p != pitch)
    {
        int hofs[HARRIS_SIZE * HARRIS_SIZE];
        for (int i = 0; i < HARRIS_SIZE; i++)
        {
            for (int j = 0; j < HARRIS_SIZE; j++)
                hofs[i * HARRIS_SIZE + j] = i * pitch + j;
        }
        CHECK(cudaMemcpyToSymbol(ofs, hofs, HARRIS_SIZE * HARRIS_SIZE * sizeof(int), 0, cudaMemcpyHostToDevice));
        p = pitch;
    }
}

void makeOffsets(int* pitchs, int noctaves)
{
    const int offsets[16][2] = {
        {0,  3}, { 1,  3}, { 2,  2}, { 3,  1}, { 3, 0}, { 3, -1}, { 2, -2}, { 1, -3},
        {0, -3}, {-1, -3}, {-2, -2}, {-3, -1}, {-3, 0}, {-3,  1}, {-2,  2}, {-1,  3}
    };

    int* hpixel = new int[25 * noctaves];
    int* temp_pixel = hpixel;
    for (int i = 0; i < noctaves; i++)
    {
        int k = 0;
        for (; k < FAST_PATTERN; k++)
            temp_pixel[k] = offsets[k][0] + offsets[k][1] * pitchs[i];
        for (; k < 25; k++)
            temp_pixel[k] = temp_pixel[k - FAST_PATTERN];
        temp_pixel += 25;
    }
    CHECK(cudaMemcpyToSymbol(dpixel, hpixel, 25 * noctaves * sizeof(int), 0, cudaMemcpyHostToDevice));
    delete[] hpixel;
}

__global__ void gDownSampleUnroll4(unsigned char* src, unsigned char* dst, int factor, 
                                    int dw, int dh, int dp, int sp)
{
    int yi = blockIdx.y * X2 + threadIdx.y;
    if (yi >= dh) return;

    int ds = yi * dp;
    int ss = yi * sp;
    int xi = blockIdx.x * X2 * 4 + threadIdx.x;
#pragma unroll
    for (int i = 0; i < 4; i++)
    {
        if (xi >= dw) return;
        dst[ds + xi] = src[(ss + xi) << factor];
        xi += X2;
    }
}

__device__ float dHarrisScore(const unsigned char* ptr, const int pitch)
{
    int dx = 0, dy = 0, dxx = 0, dyy = 0, dxy = 0;
    const unsigned char* temp_ptr = ptr + (-HARRIS_SIZE / 2) * pitch + (-HARRIS_SIZE / 2);
    for (int i = 0; i < HARRIS_SIZE * HARRIS_SIZE; i++)
    {
        const unsigned char* curr_ptr = temp_ptr + ofs[i];
        dx = ((int)curr_ptr[1] - (int)curr_ptr[-1]) * 2 +
             ((int)curr_ptr[-pitch + 1] - (int)curr_ptr[-pitch - 1]) +
             ((int)curr_ptr[pitch + 1] - (int)curr_ptr[pitch - 1]);
        dy = ((int)curr_ptr[pitch] - (int)curr_ptr[-pitch]) * 2 +
             ((int)curr_ptr[pitch - 1] - (int)curr_ptr[-pitch - 1]) +
             ((int)curr_ptr[pitch + 1] - (int)curr_ptr[-pitch + 1]);
        dxx += dx * dx;
        dyy += dy * dy;
        dxy += dx * dy;
    }
    int dsxy = dxx + dyy;
    return (dxx * dyy - dxy * dxy - HARRIS_K * (dsxy * dsxy)) * d_scale_sq_sq;
}

__device__ float dFastScore(const unsigned char* ptr, const int* pixel, int threshold)
{
    int k, v = ptr[0];
    short d[N];
    for (k = 0; k < N; k++)
        d[k] = (short)(v - ptr[pixel[k]]);

    int a0 = threshold;
    for (k = 0; k < 16; k += 2)
    {
        int a = min((int)d[k + 1], (int)d[k + 2]);
        a = min(a, (int)d[k + 3]);
        if (a <= a0) continue;
        a = min(a, (int)d[k + 4]);
        a = min(a, (int)d[k + 5]);
        a = min(a, (int)d[k + 6]);
        a = min(a, (int)d[k + 7]);
        a = min(a, (int)d[k + 8]);
        a0 = max(a0, min(a, (int)d[k]));
        a0 = max(a0, min(a, (int)d[k + 9]));
    }

    int b0 = -a0;
    for (k = 0; k < 16; k += 2)
    {
        int b = max((int)d[k + 1], (int)d[k + 2]);
        b = max(b, (int)d[k + 3]);
        b = max(b, (int)d[k + 4]);
        b = max(b, (int)d[k + 5]);
        if (b >= b0) continue;
        b = max(b, (int)d[k + 6]);
        b = max(b, (int)d[k + 7]);
        b = max(b, (int)d[k + 8]);
        b0 = min(b0, max(b, (int)d[k]));
        b0 = min(b0, max(b, (int)d[k + 9]));
    }

    threshold = -b0 - 1;
    return threshold;
}

__global__ void gCalcExtramaMap(unsigned char* image, float* vmap, int* layer_map, 
                                 int threshold, int octave, bool harris_score, 
                                 int w, int h, int p, int p0)
{
    int ix = blockIdx.x * X2 + threadIdx.x + 3;
    int iy = blockIdx.y * X2 + threadIdx.y + 3;
    if (ix >= w - 3 || iy >= h - 3) return;

    int idx = iy * p + ix;
    const unsigned char* ptr = image + idx;
    const unsigned char* tab = dthresh_table + 255 - ptr[0];
    const int* odpixel = dpixel + 25 * octave;
    
    int d = tab[ptr[odpixel[0]]] | tab[ptr[odpixel[8]]];
    if (d == 0) return;

    d &= tab[ptr[odpixel[2]]] | tab[ptr[odpixel[10]]];
    d &= tab[ptr[odpixel[4]]] | tab[ptr[odpixel[12]]];
    d &= tab[ptr[odpixel[6]]] | tab[ptr[odpixel[14]]];
    if (d == 0) return;

    d &= tab[ptr[odpixel[1]]] | tab[ptr[odpixel[9]]];
    d &= tab[ptr[odpixel[3]]] | tab[ptr[odpixel[11]]];
    d &= tab[ptr[odpixel[5]]] | tab[ptr[odpixel[13]]];
    d &= tab[ptr[odpixel[7]]] | tab[ptr[odpixel[15]]];

    bool is_corner = false;
    if (d & 1)
    {
        int vt = ptr[0] - threshold, count = 0;
        for (int k = 0; k < N; k++)
        {
            if (ptr[odpixel[k]] < vt)
            {
                if (++count > K)
                {
                    is_corner = true;
                    break;
                }
            }
            else count = 0;
        }
    }
    else if (d & 2)
    {
        int vt = ptr[0] + threshold, count = 0;
        for (int k = 0; k < N; k++)
        {
            if (ptr[odpixel[k]] > vt)
            {
                if (++count > K)
                {
                    is_corner = true;
                    break;
                }
            }
            else count = 0;
        }
    }

    if (is_corner)
    {
        int x = ix << octave;
        int y = iy << octave;
        idx = y * p0 + x;
        float score = harris_score ? dHarrisScore(ptr, p) : dFastScore(ptr, odpixel, threshold);
        if (vmap[idx] < score)
        {
            vmap[idx] = score;
            layer_map[idx] = octave;
        }
    }
}

__global__ void gNmsUnroll4(OrbPoint* points, float* vmap, int* layer_map, 
                            int border, int w, int h, int p)
{
    int iy = blockIdx.y * X2 + threadIdx.y + border;
    if (iy >= h - border) return;
    
    int ystart = iy * p;
    int ix = blockIdx.x * X2 * 4 + threadIdx.x + border;
#pragma unroll
    for (int i = 0; i < 4; i++)
    {
        if (ix >= w - border) return;

        int vidx = ystart + ix;
        if (vmap[vidx] > 0 && d_point_counter < d_max_num_points)
        {
            int o = layer_map[vidx];
            int r = 1 << o;
            int new_ystart = (iy - r) * p;
            bool to_nms = false;
            
            for (int dy = -r; dy <= r && !to_nms; dy++)
            {
                int new_vidx = new_ystart + ix - r;
                for (int dx = -r; dx <= r; dx++)
                {
                    if ((dx != 0 || dy != 0) &&
                        (vmap[new_vidx] > 0 &&
                         (vmap[new_vidx] > vmap[vidx] ||
                          (vmap[new_vidx] == vmap[vidx] && dx <= 0 && dy <= 0))))
                    {
                        to_nms = true;
                        break;
                    }
                    new_vidx++;
                }
                new_ystart += p;
            }

            if (!to_nms && d_point_counter < d_max_num_points)
            {
                unsigned int pi = atomicInc(&d_point_counter, 0x7fffffff);
                if (pi < d_max_num_points)
                {
                    points[pi].x = ix;
                    points[pi].y = iy;
                    points[pi].octave = o;
                    points[pi].score = vmap[vidx];
                }
            }
        }
        ix += X2;
    }
}

__device__ __inline__ float dFastAtan2(float y, float x)
{
    const float absx = fabs(x);
    const float absy = fabs(y);
    const float a = __fdiv_rn(min(absx, absy), max(absx, absy));
    const float s = a * a;
    float r = __fmaf_rn(__fmaf_rn(__fmaf_rn(-0.0464964749f, s, 0.15931422f), s, -0.327622764f), s * a, a);
    r = (absy > absx ? H_PI - r : r);
    r = (x < 0 ? M_PI - r : r);
    r = (y < 0 ? -r : r);
    return r;
}

__global__ void angleIC(unsigned char* images, OrbPoint* points, int half_k, int noctaves)
{
    unsigned int pi = blockIdx.x * X1 + threadIdx.x;
    if (pi < d_point_counter)
    {
        OrbPoint* p = points + pi;
        
        if (p->octave < 0 || p->octave >= noctaves) {
            p->angle = 0.0f;
            return;
        }
        
        const int pitch = angle_param[p->octave];
        const int moffset = angle_param[p->octave + noctaves];
        
        if (pitch <= 0 || moffset < 0) {
            p->angle = 0.0f;
            return;
        }
        
        const unsigned char* center = images + moffset + (p->y >> p->octave) * pitch + (p->x >> p->octave);

        int m_01 = 0, m_10 = 0;
        for (int u = -half_k; u <= half_k; ++u)
            m_10 += u * center[u];

        int v_sum = 0, pofs = 0, val_plus = 0, val_minus = 0;
        for (int v = 1; v <= half_k; ++v)
        {
            v_sum = 0;
            pofs = v * pitch;
            for (int u = -d_umax[v]; u <= d_umax[v]; ++u)
            {
                val_plus = center[u + pofs];
                val_minus = center[u - pofs];
                v_sum += (val_plus - val_minus);
                m_10 += u * (val_plus + val_minus);
            }
            m_01 += v * v_sum;
        }
        p->angle = dFastAtan2(__int2float_rn(m_01), __int2float_rn(m_10));
    }
}

__global__ void gConv2dUnroll(unsigned char* src, unsigned char* dst, int width, int height, int pitch)
{
    __shared__ float xrows[R4][X2];
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int xp = blockIdx.x * DX + tx;
    const int yp = blockIdx.y * X2 + ty;
    float* k = d_gauss;
    int xs[R2 + 1] = { 0 };
    for (int i = -GR; i <= GR; i++)
    {
        xs[i + GR] = dealBorder(xp + i, width);
    }

#pragma unroll
    for (int l = -R2; l < GR; l += GR)
    {
        int ly = l + ty;
        int ystart = dealBorder(yp + l + GR, height) * pitch;
        float wsum = k[0] * (int)src[ystart + xs[GR]];
        for (int i = 1; i <= GR; i++)
        {
            wsum += k[i] * ((int)src[ystart + xs[GR - i]] + (int)src[ystart + xs[GR + i]]);
        }
        xrows[ly + R2][tx] = wsum;
    }
    __syncthreads();

#pragma unroll
    for (int l = GR; l < X2; l += GR)
    {
        int ly = l + ty;
        int ystart = dealBorder(yp + l + GR, height) * pitch;
        float wsum = k[0] * (int)src[ystart + xs[GR]];
        for (int i = 1; i <= GR; i++)
        {
            wsum += k[i] * ((int)src[ystart + xs[GR - i]] + (int)src[ystart + xs[GR + i]]);
        }
        xrows[(ly + R2) % R4][tx] = wsum;

        int ys = yp + l - GR;
        if (xp < width && ys < height && tx < DX)
        {
            wsum = k[0] * xrows[(ly + 0) % R4][tx];
            for (int i = 1; i <= GR; i++)
            {
                wsum += k[i] * (xrows[(ly - i) % R4][tx] + xrows[(ly + i) % R4][tx]);
            }
            dst[ys * pitch + xp] = (unsigned char)__float2int_rn(wsum);
        }
        __syncthreads();
    }

    int l = X2 % GR == 0 ? X2 - GR : X2 / GR * GR;
    int ys = yp + l;
    if (xp < width && ys < height && tx < DX)
    {
        int ly = ty + l + GR;
        float wsum = k[0] * xrows[(ly + 0) % R4][tx];
        for (int i = 1; i <= GR; i++)
        {
            wsum += k[i] * (xrows[(ly - i) % R4][tx] + xrows[(ly + i) % R4][tx]);
        }
        dst[ys * pitch + xp] = (unsigned char)__float2int_rn(wsum);
    }
}

__device__ __inline__ unsigned char getValue(const unsigned char* center, const int2* pattern, 
                                              float sine, float cose, int pitch, int idx)
{
    const float x = pattern[idx].x;
    const float y = pattern[idx].y;
    const int ix = __float2int_rn(x * cose - y * sine);
    const int iy = __float2int_rn(x * sine + y * cose);
    return *(center + iy * pitch + ix);
}

__device__ __inline__ unsigned char feature2(const unsigned char* center, const int2* pattern, 
                                              float sine, float cose, int pitch)
{
    return (getValue(center, pattern, sine, cose, pitch, 0) < getValue(center, pattern, sine, cose, pitch, 1)) |
           ((getValue(center, pattern, sine, cose, pitch, 2) < getValue(center, pattern, sine, cose, pitch, 3)) << 1) |
           ((getValue(center, pattern, sine, cose, pitch, 4) < getValue(center, pattern, sine, cose, pitch, 5)) << 2) |
           ((getValue(center, pattern, sine, cose, pitch, 6) < getValue(center, pattern, sine, cose, pitch, 7)) << 3) |
           ((getValue(center, pattern, sine, cose, pitch, 8) < getValue(center, pattern, sine, cose, pitch, 9)) << 4) |
           ((getValue(center, pattern, sine, cose, pitch, 10) < getValue(center, pattern, sine, cose, pitch, 11)) << 5) |
           ((getValue(center, pattern, sine, cose, pitch, 12) < getValue(center, pattern, sine, cose, pitch, 13)) << 6) |
           ((getValue(center, pattern, sine, cose, pitch, 14) < getValue(center, pattern, sine, cose, pitch, 15)) << 7);
}

__global__ void gDescrible(unsigned char* images, OrbPoint* points, unsigned char* desc, 
                            int wta_k, int noctaves)
{
    unsigned int pi = blockIdx.x;
    if (pi >= d_point_counter) return;

    OrbPoint* p = &points[pi];
    
    if (p->octave < 0 || p->octave >= noctaves)
        return;
    
    const float sine = __sinf(p->angle);
    const float cose = __cosf(p->angle);
    const int pitch = angle_param[p->octave];
    const int moffset = angle_param[p->octave + noctaves];
    
    if (pitch <= 0 || moffset < 0)
        return;
    const unsigned char* center = images + moffset + (p->y >> p->octave) * pitch + (p->x >> p->octave);
    unsigned char* curr_desc = desc + pi * 32;

    unsigned int tix = threadIdx.x;
    const int pstep = (wta_k == 2 || wta_k == 4) ? 16 : 12;
    const int2* pattern = d_pattern + pstep * tix;
    curr_desc[tix] = feature2(center, pattern, sine, cose, pitch);
}

__global__ void gHammingMatch(OrbPoint* points1, unsigned char* desc1, unsigned char* desc2, int n1, int n2)
{
    __shared__ int idx_1st[X2];
    __shared__ int idx_2nd[X2];
    __shared__ int score_1st[X2];
    __shared__ int score_2nd[X2];

    int pi1 = blockIdx.x;
    int tix = threadIdx.x;

    idx_1st[tix] = 0;
    idx_2nd[tix] = 0;
    score_1st[tix] = 256;
    score_2nd[tix] = 256;
    __syncthreads();

    unsigned long long* d1i = (unsigned long long*)(desc1 + 32 * pi1);
    for (int i = 0; i < n2; i += X2)
    {
        int pi2 = i + tix;
        unsigned long long* d2i = (unsigned long long*)(desc2 + 32 * pi2);
        if (pi2 < n2)
        {
            int score = 0;
#pragma unroll
            for (int j = 0; j < 4; ++j)
            {
                score += __popcll(d1i[j] ^ d2i[j]);
            }
            if (score < score_1st[tix])
            {
                score_2nd[tix] = score_1st[tix];
                score_1st[tix] = score;
                idx_2nd[tix] = idx_1st[tix];
                idx_1st[tix] = pi2;
            }
            else if (score < score_2nd[tix])
            {
                score_2nd[tix] = score;
                idx_2nd[tix] = pi2;
            }
        }
    }
    __syncthreads();

    for (int i = X2 / 2; i >= 1; i /= 2)
    {
        if (tix < i)
        {
            int nix = tix + i;
            if (score_1st[nix] < score_1st[tix])
            {
                score_2nd[tix] = score_1st[tix];
                score_1st[tix] = score_1st[nix];
                idx_2nd[tix] = idx_1st[tix];
                idx_1st[tix] = idx_1st[nix];
            }
            else if (score_1st[nix] < score_2nd[tix])
            {
                score_2nd[tix] = score_1st[nix];
                idx_2nd[tix] = idx_1st[nix];
            }
            if (score_2nd[nix] < score_2nd[tix])
            {
                score_2nd[tix] = score_2nd[nix];
                idx_2nd[tix] = idx_2nd[nix];
            }
        }
        __syncthreads();
    }

    if (tix == 0)
    {
        OrbPoint* pt1 = points1 + pi1;
        if (score_1st[0] * 4 < score_2nd[0] * 3 && score_1st[0] < MAX_DIST)
        {
            pt1->match = idx_1st[0];
            pt1->distance = score_1st[0];
        }
        else
        {
            pt1->match = -1;
            pt1->distance = -1;
        }
    }
}

void hFastDectectWithNMS(unsigned char* image, unsigned char* octave_images, float* vmem, 
                          OrbData& result, int* oszp, int noctaves, int threshold, 
                          int border, bool harris_score)
{
    if (border < 3) border = 3;
    
    int* osizes = oszp;
    int* widths = osizes + noctaves;
    int* heights = widths + noctaves;
    int* pitchs = heights + noctaves;
    int* offsets = pitchs + noctaves;

    float* vmap = vmem;
    int* layer_map = (int*)(vmap + osizes[0]);

    dim3 block(X2, X2);
    dim3 grid1;

    CHECK(cudaMemcpy(octave_images, image, osizes[0] * sizeof(unsigned char), cudaMemcpyDeviceToDevice));

    int factor = 1;
    for (int i = 1; i < noctaves; i++)
    {
        grid1.x = (widths[i] + X2 * 4 - 1) / (X2 * 4);
        grid1.y = (heights[i] + X2 - 1) / X2;
        gDownSampleUnroll4<<<grid1, block>>>(image, octave_images + offsets[i], factor, 
                                              widths[i], heights[i], pitchs[i], pitchs[0]);
        CHECK(cudaDeviceSynchronize());
        factor++;
    }

    dim3 grid2;
    for (int i = 0; i < noctaves; i++)
    {
        if (harris_score)
            setHarrisOffsets(pitchs[i]);
        grid2.x = (widths[i] - 6 + X2 - 1) / X2;
        grid2.y = (heights[i] - 6 + X2 - 1) / X2;
        gCalcExtramaMap<<<grid2, block>>>(octave_images + offsets[i], vmap, layer_map, threshold, i,
                                           harris_score, widths[i], heights[i], pitchs[i], pitchs[0]);
        CHECK(cudaDeviceSynchronize());
    }

    int total_border = border + border;
    dim3 grid3((widths[0] - total_border + X2 * 4 - 1) / (X2 * 4), 
               (heights[0] - total_border + X2 - 1) / X2);
    gNmsUnroll4<<<grid3, block>>>(result.d_data, vmap, layer_map, border, 
                                   widths[0], heights[0], pitchs[0]);
    CHECK(cudaDeviceSynchronize());
    CheckMsg("hFastDectectWithNMS() execution failed!\n");
}

void hComputeAngle(unsigned char* octave_images, OrbData& result, int* oszp, int noctaves, int patch_size)
{
    if (result.num_pts <= 0) return;
    
    int* aparams = oszp + noctaves * 3;
    CHECK(cudaMemcpyToSymbol(angle_param, aparams, noctaves * 2 * sizeof(int), 0, cudaMemcpyHostToDevice));

    dim3 block(X1);
    dim3 grid((result.num_pts + X1 - 1) / X1);
    angleIC<<<grid, block>>>(octave_images, result.d_data, patch_size / 2, noctaves);
    CHECK(cudaDeviceSynchronize());
    CheckMsg("hComputeAngle() execution failed!\n");
}

void hGassianBlur(unsigned char* octave_images, int* oszp, int noctaves)
{
    int* osizes = oszp;
    int* widths = osizes + noctaves;
    int* heights = widths + noctaves;
    int* pitchs = heights + noctaves;
    int* offsets = pitchs + noctaves;

    dim3 block(X2, GR), grid;
    for (int i = 0; i < noctaves; i++)
    {
        unsigned char* mem = octave_images + offsets[i];
        grid.x = (widths[i] + DX - 1) / DX;
        grid.y = (heights[i] + X2 - 1) / X2;
        gConv2dUnroll<<<grid, block>>>(mem, mem, widths[i], heights[i], pitchs[i]);
    }
    CHECK(cudaDeviceSynchronize());
    CheckMsg("hGassianBlur() execution failed!\n");
}

void hDescribe(unsigned char* octave_images, OrbData& result, unsigned char* desc, int wta_k, int noctaves)
{
    if (result.num_pts <= 0) return;
    
    dim3 block(X2);
    dim3 grid(result.num_pts);
    gDescrible<<<grid, block>>>(octave_images, result.d_data, desc, wta_k, noctaves);
    CHECK(cudaDeviceSynchronize());
    CheckMsg("hDescribe() execution failed!\n");
}

void hMatch(OrbData& result1, OrbData& result2, unsigned char* desc1, unsigned char* desc2)
{
    if (result1.num_pts <= 0 || result2.num_pts <= 0) return;
    
    dim3 block(X2);
    dim3 grid(result1.num_pts);
    gHammingMatch<<<grid, block>>>(result1.d_data, desc1, desc2, result1.num_pts, result2.num_pts);
    CHECK(cudaDeviceSynchronize());
    CheckMsg("hMatch() execution failed\n");
}

}
