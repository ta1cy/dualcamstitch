#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <thread>
#include <mutex>
#include <atomic>
#include <iostream>
#include <algorithm>
#include <cmath>
#include <vector>
#include <cstdio>

class LatestFrameGrabber {
public:
    LatestFrameGrabber(cv::VideoCapture& cap) : cap_(cap), stopped_(false) {
        thread_ = std::thread(&LatestFrameGrabber::loop, this);
    }

    ~LatestFrameGrabber() {
        stop();
    }

    void loop() {
        cv::Mat frm;
        while (!stopped_) {
            if (cap_.read(frm) && !frm.empty()) {
                std::lock_guard<std::mutex> lock(mutex_);
                frame_ = frm.clone();
            }
            std::this_thread::sleep_for(std::chrono::microseconds(500));
        }
    }

    bool read(cv::Mat& out) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (frame_.empty())
            return false;
        out = frame_.clone();
        return true;
    }

    void stop() {
        stopped_ = true;
        if (thread_.joinable())
            thread_.join();
    }

    void restart() {
        stopped_ = false;
        frame_ = cv::Mat();
        thread_ = std::thread(&LatestFrameGrabber::loop, this);
    }

private:
    cv::VideoCapture& cap_;
    cv::Mat frame_;
    std::mutex mutex_;
    std::atomic<bool> stopped_;
    std::thread thread_;
};

cv::VideoCapture openCamera(const std::string& dev, int width, int height, int fps) {
    cv::VideoCapture cap(dev, cv::CAP_V4L2);
    /*
    if (!cap.isOpened()) {
        cap.open(index, cv::CAP_V4L2);
    }
    */

    if (!cap.isOpened())
        throw std::runtime_error("Failed to open camera index " + dev);
    cap.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));
    cap.set(cv::CAP_PROP_FRAME_WIDTH, width);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, height);
    cap.set(cv::CAP_PROP_FPS, fps);
    cap.set(cv::CAP_PROP_BUFFERSIZE, 1);

    return cap;
}

void resizeToCommon(const cv::Mat& img0, const cv::Mat& img1, cv::Mat& out0, cv::Mat& out1) {
    int h = std::min(img0.rows, img1.rows);
    int w = std::min(img0.cols, img1.cols);
    cv::resize(img0, out0, cv::Size(w, h), 0, 0, cv::INTER_AREA);
    cv::resize(img1, out1, cv::Size(w, h), 0, 0, cv::INTER_AREA);
}

cv::Mat sideBySide(const cv::Mat& img0, const cv::Mat& img1) {
    cv::Mat result;
    cv::hconcat(img0, img1, result);
    return result;
}

struct ColorCorrection {
    cv::Vec3f gain;
    cv::Vec3f offset;
    
    ColorCorrection() : gain(1.0f, 1.0f, 1.0f), offset(0.0f, 0.0f, 0.0f) {}
};

struct ShiftTimingInfo {
    double detect_ms = 0;
    double match_ms = 0;
    int kp0 = 0;
    int kp1 = 0;
    int good_matches = 0;
};

void computeColorCorrection(const cv::Mat& img0, const cv::Mat& img1, int dx,
                             ColorCorrection& correction, float strength = 0.8f) {
    int w0 = img0.cols;
    int w1 = img1.cols;
    int h = img0.rows;
    
    int overlapStart = std::max(0, -dx);
    int overlapEnd = std::min(w1, w0 - dx);
    int overlapLen = overlapEnd - overlapStart;
    
    if (overlapLen < 20) {
        correction = ColorCorrection();
        return;
    }
    
    int margin = overlapLen / 4;
    int sampleStart = overlapStart + margin;
    int sampleEnd = overlapEnd - margin;
    int sampleWidth = sampleEnd - sampleStart;
    
    if (sampleWidth < 10) {
        correction = ColorCorrection();
        return;
    }
    
    cv::Rect roi0(sampleStart + dx, 0, sampleWidth, h);
    cv::Rect roi1(sampleStart, 0, sampleWidth, h);
    
    roi0.x = std::clamp(roi0.x, 0, w0 - 1);
    roi0.width = std::min(roi0.width, w0 - roi0.x);
    roi1.x = std::clamp(roi1.x, 0, w1 - 1);
    roi1.width = std::min(roi1.width, w1 - roi1.x);
    
    if (roi0.width < 10 || roi1.width < 10) {
        correction = ColorCorrection();
        return;
    }
    
    int commonWidth = std::min(roi0.width, roi1.width);
    roi0.width = commonWidth;
    roi1.width = commonWidth;
    
    cv::Mat patch0 = img0(roi0);
    cv::Mat patch1 = img1(roi1);
    
    cv::Scalar mean0 = cv::mean(patch0);
    cv::Scalar mean1 = cv::mean(patch1);
    
    cv::Scalar stddev0, stddev1;
    cv::meanStdDev(patch0, cv::Scalar(), stddev0);
    cv::meanStdDev(patch1, cv::Scalar(), stddev1);
    
    for (int c = 0; c < 3; c++) {
        float m0 = static_cast<float>(mean0[c]);
        float m1 = static_cast<float>(mean1[c]);
        float s0 = static_cast<float>(stddev0[c]);
        float s1 = static_cast<float>(stddev1[c]);
        
        float gain = 1.0f;
        if (s1 > 1.0f && s0 > 1.0f) {
            gain = s0 / s1;
            gain = std::clamp(gain, 0.5f, 2.0f);
        }
        
        float offset = m0 - gain * m1;
        offset = std::clamp(offset, -50.0f, 50.0f);
        
        correction.gain[c] = 1.0f + strength * (gain - 1.0f);
        correction.offset[c] = strength * offset;
    }
}

void applyColorCorrection(const cv::Mat& src, cv::Mat& dst, const ColorCorrection& correction) {
    dst.create(src.rows, src.cols, src.type());
    
    for (int y = 0; y < src.rows; y++) {
        const uchar* srcRow = src.ptr<uchar>(y);
        uchar* dstRow = dst.ptr<uchar>(y);
        
        for (int x = 0; x < src.cols; x++) {
            int idx = x * 3;
            for (int c = 0; c < 3; c++) {
                float val = correction.gain[c] * srcRow[idx + c] + correction.offset[c];
                dstRow[idx + c] = static_cast<uchar>(std::clamp(val, 0.0f, 255.0f));
            }
        }
    }
}

float computeOptimalSeamFrac(int dx, int w0, int w1, float margin = 0.05f) {
    int overlapStart = std::max(0, -dx);
    int overlapEnd = std::min(w1, w0 - dx);
    int overlapLen = overlapEnd - overlapStart;

    if (overlapLen <= 0)
        return 0.5f;

    int outputWidth = w0 + w1 - overlapLen;
    int targetSeamX = outputWidth / 2;

    int seamX0 = targetSeamX;
    int seamX1 = seamX0 - dx;

    float seamFrac = static_cast<float>(seamX1 - overlapStart) / static_cast<float>(overlapLen);

    seamFrac = std::clamp(seamFrac, margin, 1.0f - margin);

    return seamFrac;
}

bool estimateHorizontalShift(const cv::Mat& img0, const cv::Mat& img1,
                              int& dx, int& agree,
                              int maxFeatures = 1500, float ratio = 0.75f, bool debug = false,
                              ShiftTimingInfo* timing = nullptr) {
    cv::Mat g0, g1;
    cv::cvtColor(img0, g0, cv::COLOR_BGR2GRAY);
    cv::cvtColor(img1, g1, cv::COLOR_BGR2GRAY);

    auto orb = cv::ORB::create(maxFeatures, 1.2f, 8, 31, 0, 2, cv::ORB::HARRIS_SCORE, 31, 10);

    std::vector<cv::KeyPoint> k0, k1;
    cv::Mat d0, d1;

    auto t_det_start = std::chrono::high_resolution_clock::now();
    orb->detectAndCompute(g0, cv::noArray(), k0, d0);
    orb->detectAndCompute(g1, cv::noArray(), k1, d1);
    auto t_det_end = std::chrono::high_resolution_clock::now();

    if (timing) {
        timing->detect_ms = std::chrono::duration<double, std::milli>(t_det_end - t_det_start).count();
        timing->kp0 = static_cast<int>(k0.size());
        timing->kp1 = static_cast<int>(k1.size());
    }

    if (d0.empty() || d1.empty() || k0.size() < 12 || k1.size() < 12)
        return false;

    cv::BFMatcher matcher(cv::NORM_HAMMING, false);
    std::vector<std::vector<cv::DMatch>> knn;

    auto t_match_start = std::chrono::high_resolution_clock::now();
    matcher.knnMatch(d0, d1, knn, 2);

    std::vector<cv::DMatch> good;
    for (const auto& pair : knn) {
        if (pair.size() < 2) continue;
        if (pair[0].distance < ratio * pair[1].distance)
            good.push_back(pair[0]);
    }
    auto t_match_end = std::chrono::high_resolution_clock::now();

    if (timing) {
        timing->match_ms = std::chrono::duration<double, std::milli>(t_match_end - t_match_start).count();
        timing->good_matches = static_cast<int>(good.size());
    }

    if (good.size() < 12)
        return false;

    std::vector<float> dxs;
    for (const auto& m : good) {
        float x0 = k0[m.queryIdx].pt.x;
        float x1 = k1[m.trainIdx].pt.x;
        dxs.push_back(x0 - x1);
    }

    std::sort(dxs.begin(), dxs.end());
    float median = dxs[dxs.size() / 2];
    dx = static_cast<int>(median);

    float tol = 6.0f;
    agree = 0;
    for (float d : dxs) {
        if (std::abs(d - median) <= tol)
            agree++;
    }

    if (debug)
        std::cout << "[shift] good=" << good.size() << " dx=" << dx << " agree=" << agree << std::endl;

    return true;
}

cv::Mat joinMiddleSeamBlended(const cv::Mat& img0, const cv::Mat& img1, int dx, float seamFrac,
                               int& seamOutX, int& seamX1, int blendWidth = 50) {
    int h = img0.rows;
    int w0 = img0.cols;
    int w1 = img1.cols;

    int overlapStart = std::max(0, -dx);
    int overlapEnd = std::min(w1, w0 - dx);

    overlapStart = std::clamp(overlapStart, 0, w1);
    overlapEnd = std::clamp(overlapEnd, overlapStart, w1);

    int overlapLen = overlapEnd - overlapStart;
    if (overlapLen <= 0) {
        cv::Mat out;
        cv::hconcat(img0, img1, out);
        seamOutX = w0;
        seamX1 = w0;
        return out;
    }

    seamFrac = std::clamp(seamFrac, 0.0f, 1.0f);
    seamX1 = overlapStart + static_cast<int>(std::round(seamFrac * overlapLen));
    seamX1 = std::clamp(seamX1, 1, w1 - 1);

    int seamX0 = seamX1 + dx;
    seamX0 = std::clamp(seamX0, 1, w0 - 1);

    int halfBlend = blendWidth / 2;
    int blendStart0 = std::max(1, seamX0 - halfBlend);
    int blendEnd0 = std::min(w0 - 1, seamX0 + halfBlend);
    int actualBlendWidth = blendEnd0 - blendStart0;

    if (actualBlendWidth < 4) {
        cv::Mat left = img0(cv::Rect(0, 0, seamX0, h));
        cv::Mat right = img1(cv::Rect(seamX1, 0, w1 - seamX1, h));
        cv::Mat out;
        cv::hconcat(left, right, out);
        seamOutX = seamX0;
        return out;
    }

    cv::Mat left = img0(cv::Rect(0, 0, blendStart0, h));

    int blendStart1 = blendStart0 - dx;
    int blendEnd1 = blendEnd0 - dx;
    blendStart1 = std::clamp(blendStart1, 0, w1 - 1);
    blendEnd1 = std::clamp(blendEnd1, blendStart1 + 1, w1);
    int blendLen1 = blendEnd1 - blendStart1;

    cv::Mat blend0 = img0(cv::Rect(blendStart0, 0, actualBlendWidth, h));
    cv::Mat blend1;
    if (blendLen1 > 0 && blendLen1 == actualBlendWidth)
        blend1 = img1(cv::Rect(blendStart1, 0, blendLen1, h));
    else
        cv::resize(img1(cv::Rect(blendStart1, 0, std::max(1, blendLen1), h)), 
                   blend1, cv::Size(actualBlendWidth, h));

    cv::Mat blended(h, actualBlendWidth, img0.type());
    for (int x = 0; x < actualBlendWidth; x++) {
        float alpha = static_cast<float>(x) / static_cast<float>(actualBlendWidth - 1);
        for (int y = 0; y < h; y++) {
            cv::Vec3b p0 = blend0.at<cv::Vec3b>(y, x);
            cv::Vec3b p1 = blend1.at<cv::Vec3b>(y, x);
            blended.at<cv::Vec3b>(y, x) = cv::Vec3b(
                static_cast<uchar>((1.0f - alpha) * p0[0] + alpha * p1[0]),
                static_cast<uchar>((1.0f - alpha) * p0[1] + alpha * p1[1]),
                static_cast<uchar>((1.0f - alpha) * p0[2] + alpha * p1[2])
            );
        }
    }

    int rightStart1 = blendEnd1;
    rightStart1 = std::clamp(rightStart1, 0, w1 - 1);
    cv::Mat right = img1(cv::Rect(rightStart1, 0, w1 - rightStart1, h));

    cv::Mat out;
    cv::hconcat(left, blended, out);
    cv::hconcat(out, right, out);

    seamOutX = blendStart0 + actualBlendWidth / 2;
    return out;
}

int main(int argc, char** argv) {
    int width = 640;
    int height = 480;
    int fps = 15;
    float previewScale = 1.0f;
    int maxFeatures = 1500;
    float ratio = 0.75f;
    int minAgree = 20;
    int recalcEvery = 0;
    float seamFrac = 0.5f;
    bool autoSeam = true;
    bool debugShift = false;
    int threads = 0;
    int blendWidth = 50;
    float smoothAlpha = 0.2f;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--width" && i + 1 < argc) width = std::stoi(argv[++i]);
        else if (arg == "--height" && i + 1 < argc) height = std::stoi(argv[++i]);
        else if (arg == "--fps" && i + 1 < argc) fps = std::stoi(argv[++i]);
        else if (arg == "--preview_scale" && i + 1 < argc) previewScale = std::stof(argv[++i]);
        else if (arg == "--max_features" && i + 1 < argc) maxFeatures = std::stoi(argv[++i]);
        else if (arg == "--ratio" && i + 1 < argc) ratio = std::stof(argv[++i]);
        else if (arg == "--min_agree" && i + 1 < argc) minAgree = std::stoi(argv[++i]);
        else if (arg == "--recalc_every" && i + 1 < argc) recalcEvery = std::stoi(argv[++i]);
        else if (arg == "--seam_frac" && i + 1 < argc) { seamFrac = std::stof(argv[++i]); autoSeam = false; }
        else if (arg == "--auto_seam") autoSeam = true;
        else if (arg == "--no_auto_seam") autoSeam = false;
        else if (arg == "--debug_shift") debugShift = true;
        else if (arg == "--threads" && i + 1 < argc) threads = std::stoi(argv[++i]);
        else if (arg == "--blend_width" && i + 1 < argc) blendWidth = std::stoi(argv[++i]);
        else if (arg == "--smooth_alpha" && i + 1 < argc) smoothAlpha = std::stof(argv[++i]);
    }

    cv::setUseOptimized(true);
    cv::setNumThreads(threads);

    // will place log.txt next to the executable
    std::string exePath(argv[0]);
    std::string exeDir = exePath.substr(0, exePath.find_last_of("/\\") + 1);
    std::string logPath = exeDir + "log.csv";
    FILE* logFile = fopen(logPath.c_str(), "w");
    if (!logFile) {
        std::cerr << "Warning: could not open " << logPath << " for writing" << std::endl;
        logFile = stderr;
    }
    fprintf(logFile, "frame,capture_ms_avg,detect_describe_ms,matching_ms,color_corr_ms_avg,blend_ms_avg,frame_ms_avg,fps,kp0,kp1,good_matches\n");
    fflush(logFile);

    // modified for linux, check camera root with "v412-ctl --list-devices"
    cv::VideoCapture cap0 = openCamera("/dev/video2", width, height, fps);
    cv::VideoCapture cap1 = openCamera("/dev/video0", width, height, fps);

    LatestFrameGrabber g0(cap0);
    LatestFrameGrabber g1(cap1);

    ColorCorrection colorCorr;
    float colorCorrStrength = 0.7f;

    std::cout << " Controls:" << std::endl;
    std::cout << " q = quit" << std::endl;
    std::cout << " r = recompute dx now" << std::endl;
    std::cout << " c = recompute color correction" << std::endl;
    std::cout << " x = reset camera streams" << std::endl;

    cv::namedWindow("cameras", cv::WINDOW_NORMAL);
    cv::namedWindow("joined", cv::WINDOW_NORMAL);

    for (int i = 0; i < 20; i++) {
        cv::Mat tmp;
        g0.read(tmp);
        g1.read(tmp);
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    int dx = 0;
    int agree = 0;
    bool hasDx = false;
    int frameCount = 0;
    float smoothedDx = 0.0f;

    cv::Mat img0, img1;
    if (g0.read(img0) && g1.read(img1)) {
        cv::Mat img0c, img1c;
        resizeToCommon(img0, img1, img0c, img1c);
        int dxTry, agreeTry;
        if (estimateHorizontalShift(img0c, img1c, dxTry, agreeTry, maxFeatures, ratio, debugShift)) {
            if (agreeTry >= minAgree) {
                dx = dxTry;
                agree = agreeTry;
                hasDx = true;
                smoothedDx = static_cast<float>(dx);
                computeColorCorrection(img0c, img1c, dx, colorCorr, colorCorrStrength);
            }
        }
    }

    std::cout << "initial dx: " << dx << " (agree=" << agree << ")" << std::endl;

    ShiftTimingInfo lastTiming;
    double acc_capture_ms = 0, acc_colorcorr_ms = 0, acc_blend_ms = 0, acc_frame_ms = 0;
    int stats_count = 0;

    auto prevTime = std::chrono::high_resolution_clock::now();

    while (true) {
        auto t_cap_start = std::chrono::high_resolution_clock::now();
        if (!g0.read(img0) || !g1.read(img1)) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            continue;
        }
        auto t_cap_end = std::chrono::high_resolution_clock::now();
        double captureMs = std::chrono::duration<double, std::milli>(t_cap_end - t_cap_start).count();

        frameCount++;

        cv::Mat img0c, img1c;
        resizeToCommon(img0, img1, img0c, img1c);
        int h = img0c.rows;
        int w = img0c.cols;

        if (recalcEvery > 0 && (frameCount % recalcEvery == 0)) {
            int dxTry, agreeTry;
            if (estimateHorizontalShift(img0c, img1c, dxTry, agreeTry, maxFeatures, ratio, debugShift, &lastTiming)) {
                if (agreeTry >= minAgree) {
                    smoothedDx = smoothAlpha * static_cast<float>(dxTry) + (1.0f - smoothAlpha) * smoothedDx;
                    dx = static_cast<int>(std::round(smoothedDx));
                    agree = agreeTry;
                    hasDx = true;
                }
            }
            if (hasDx && frameCount % (recalcEvery * 30) == 0)
                computeColorCorrection(img0c, img1c, dx, colorCorr, colorCorrStrength);
        }

        if (autoSeam && hasDx)
            seamFrac = computeOptimalSeamFrac(dx, w, w);

        cv::Mat cams = sideBySide(img0c, img1c);

        cv::Mat out;
        int seamX = w;
        int seamX1 = -1;

        auto blendStart = std::chrono::high_resolution_clock::now();
        if (hasDx) {
            cv::Mat img1Corrected;
            auto t_cc_start = std::chrono::high_resolution_clock::now();
            applyColorCorrection(img1c, img1Corrected, colorCorr);
            auto t_cc_end = std::chrono::high_resolution_clock::now();
            double ccMs = std::chrono::duration<double, std::milli>(t_cc_end - t_cc_start).count();
            acc_colorcorr_ms += ccMs;
            out = joinMiddleSeamBlended(img0c, img1Corrected, dx, seamFrac, seamX, seamX1, blendWidth);
        } else
            out = sideBySide(img0c, img1c);

        auto blendEnd = std::chrono::high_resolution_clock::now();
        double blendMs = std::chrono::duration<double, std::milli>(blendEnd - blendStart).count();
        auto now = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration<double>(now - prevTime).count();
        double fpsVal = 1.0 / std::max(1e-6, elapsed);
        double frameMs = elapsed * 1000.0;
        prevTime = now;
        acc_capture_ms += captureMs;
        acc_blend_ms += blendMs;
        acc_frame_ms += frameMs;
        stats_count++;

        if (stats_count >= 10) {
            double avg_cap = acc_capture_ms / stats_count;
            double avg_cc = acc_colorcorr_ms / stats_count;
            double avg_blend = acc_blend_ms / stats_count;
            double avg_frame = acc_frame_ms / stats_count;
            double avg_fps = 1000.0 * stats_count / acc_frame_ms;
            fprintf(logFile, "%d,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.1f,%d,%d,%d\n",
                    frameCount, avg_cap, lastTiming.detect_ms, lastTiming.match_ms,
                    avg_cc, avg_blend, avg_frame, avg_fps,
                    lastTiming.kp0, lastTiming.kp1, lastTiming.good_matches);
            fflush(logFile);
            acc_capture_ms = acc_colorcorr_ms = acc_blend_ms = acc_frame_ms = 0;
            stats_count = 0;
        }

        cv::line(out, cv::Point(seamX, 0), cv::Point(seamX, out.rows - 1), cv::Scalar(0, 255, 0), 2);

        if (seamX1 >= 0)
            cv::line(cams, cv::Point(w + seamX1, 0), cv::Point(w + seamX1, h - 1), cv::Scalar(0, 255, 0), 2);

        char textBuf[256];
        snprintf(textBuf, sizeof(textBuf), "cam0 | cam1   FPS:%.1f", fpsVal);
        cv::putText(cams, textBuf, cv::Point(20, 40), cv::FONT_HERSHEY_SIMPLEX, 1.0,
                    cv::Scalar(255, 0, 0), 2, cv::LINE_AA);

        snprintf(textBuf, sizeof(textBuf), "dx:%d  agree:%d  seam:%.2f  blend:%d %s", dx, agree, seamFrac, blendWidth, autoSeam ? "(auto)" : "");
        cv::putText(out, textBuf, cv::Point(20, 40), cv::FONT_HERSHEY_SIMPLEX, 0.9,
                    cv::Scalar(255, 0, 0), 2, cv::LINE_AA);

        cv::Mat camsShow, outShow;
        if (previewScale != 1.0f) {
            int nw = std::max(2, static_cast<int>(cams.cols * previewScale));
            int nh = std::max(2, static_cast<int>(cams.rows * previewScale));
            cv::resize(cams, camsShow, cv::Size(nw, nh), 0, 0, cv::INTER_AREA);

            nw = std::max(2, static_cast<int>(out.cols * previewScale));
            nh = std::max(2, static_cast<int>(out.rows * previewScale));
            cv::resize(out, outShow, cv::Size(nw, nh), 0, 0, cv::INTER_AREA);
        } else {
            camsShow = cams;
            outShow = out;
        }

        cv::imshow("cameras", camsShow);
        cv::imshow("joined", outShow);

        int key = cv::waitKey(1) & 0xFF;
        if (key == 'q')
            break;
        else if (key == 'r') {
            int dxTry, agreeTry;
            if (estimateHorizontalShift(img0c, img1c, dxTry, agreeTry, maxFeatures, ratio, true)) {
                if (agreeTry >= minAgree) {
                    smoothedDx = static_cast<float>(dxTry);
                    dx = dxTry;
                    agree = agreeTry;
                    hasDx = true;
                    computeColorCorrection(img0c, img1c, dx, colorCorr, colorCorrStrength);
                }
            }
            std::cout << "recompute dx: " << dx << " (agree=" << agree << ")" << std::endl;
        } else if (key == 'c') {
            if (hasDx) {
                computeColorCorrection(img0c, img1c, dx, colorCorr, colorCorrStrength);
                std::cout << "recompute color: gain=(" << colorCorr.gain[0] << "," 
                          << colorCorr.gain[1] << "," << colorCorr.gain[2] << ")" << std::endl;
            }
        } else if (key == 'x') {
            g0.stop();
            g1.stop();
            cap0.release();
            cap1.release();
            cap0 = openCamera("/dev/video2", width, height, fps);
            cap1 = openCamera("/dev/video0", width, height, fps);
            g0.restart();
            g1.restart();
            for (int i = 0; i < 20; i++) {
                cv::Mat tmp;
                g0.read(tmp);
                g1.read(tmp);
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }
            dx = 0;
            agree = 0;
            hasDx = false;
            smoothedDx = 0.0f;
            colorCorr = ColorCorrection();
            if (g0.read(img0) && g1.read(img1)) {
                cv::Mat img0r, img1r;
                resizeToCommon(img0, img1, img0r, img1r);
                int dxTry, agreeTry;
                if (estimateHorizontalShift(img0r, img1r, dxTry, agreeTry, maxFeatures, ratio, debugShift)) {
                    if (agreeTry >= minAgree) {
                        dx = dxTry;
                        agree = agreeTry;
                        hasDx = true;
                        smoothedDx = static_cast<float>(dx);
                        computeColorCorrection(img0r, img1r, dx, colorCorr, colorCorrStrength);
                    }
                }
            }
            std::cout << "Camera streams reset. dx: " << dx << " (agree=" << agree << ")" << std::endl;
        }
    }

    g0.stop();
    g1.stop();
    cap0.release();
    cap1.release();
    cv::destroyAllWindows();

    if (logFile && logFile != stderr) fclose(logFile);

    return 0;
}
