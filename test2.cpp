#include <opencv2/opencv.hpp>
#include <iostream>
#include <algorithm>
#include <cmath>
#include <vector>
#include <string>

struct Params
{
    double thresholdValue = 70.0;
    double minArea        = 20.0;
    double minRatio       = 2.5;
    double minFill        = 0.4;
    double maxAngleDiff   = 15.0;
    double maxLenDiff     = 0.4;
    double maxYDiff       = 0.5;
};

struct Options
{
    std::string color = "red";
    bool useHsv    = false;
    bool showMask  = false;
    bool debug     = false;
    bool saveVideo = false;
};

struct LightBar
{
    cv::RotatedRect rect;
    cv::Point2f top, bottom;
    float dirAngle = 0.f;
    float length = 0.f;
};

cv::Mat makeMask(const cv::Mat& frame, const Params& p, const Options& opt)
{
    cv::Mat mask;
    if (opt.useHsv)
    {
        cv::Mat hsv, m1, m2;
        cv::cvtColor(frame, hsv, cv::COLOR_BGR2HSV);
        if (opt.color == "blue")
        {
            cv::inRange(hsv, cv::Scalar(100, 80, 80), cv::Scalar(130, 255, 255), mask);
        }
        else
        {
            cv::inRange(hsv, cv::Scalar(0, 60, 60), cv::Scalar(25, 255, 255), m1);
            cv::inRange(hsv, cv::Scalar(155, 60, 60), cv::Scalar(180, 255, 255), m2);
            cv::bitwise_or(m1, m2, mask);
        }
    }
    else
    {
        std::vector<cv::Mat> ch;
        cv::split(frame, ch);
        cv::Mat diff = (opt.color == "blue") ? (ch[0] - ch[2]) : (ch[2] - ch[0]);
        cv::threshold(diff, mask, p.thresholdValue, 255, cv::THRESH_BINARY);
    }

    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
    cv::morphologyEx(mask, mask, cv::MORPH_OPEN, kernel);
    cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, kernel);
    return mask;
}

std::vector<LightBar> findLightBars(const cv::Mat& mask, const Params& p)
{
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask.clone(), contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    std::vector<LightBar> bars;
    for (const auto& c : contours)
    {
        double area = cv::contourArea(c);
        if (area < p.minArea) continue;

        cv::RotatedRect rect = cv::minAreaRect(c);
        float h = std::max(rect.size.height, rect.size.width);
        float w = std::min(rect.size.height, rect.size.width);
        if (w < 1.f || h / w < p.minRatio) continue;
        if (area / (h * w) < p.minFill) continue;

        cv::Point2f pts[4];
        rect.points(pts);
        std::sort(pts, pts + 4,
                  [](const cv::Point2f& a, const cv::Point2f& b) { return a.y < b.y; });

        LightBar bar;
        bar.rect   = rect;
        bar.top    = (pts[0] + pts[1]) * 0.5f;
        bar.bottom = (pts[2] + pts[3]) * 0.5f;
        bar.length = static_cast<float>(cv::norm(bar.top - bar.bottom));
        bar.dirAngle = static_cast<float>(
            std::atan2(bar.bottom.y - bar.top.y, bar.bottom.x - bar.top.x) * 180.0 / CV_PI);
        if (bar.dirAngle < 0.f) bar.dirAngle += 180.f;

        bars.push_back(bar);
    }
    return bars;
}

bool isPair(const LightBar& a, const LightBar& b, const Params& p)
{
    float aLen = std::max(a.length, 1e-3f);
    float bLen = std::max(b.length, 1e-3f);

    float dirDiff = std::abs(a.dirAngle - b.dirAngle);
    if (dirDiff > 90.f) dirDiff = 180.f - dirDiff;

    float lenDiff = std::abs(a.length - b.length) / std::max(aLen, bLen);
    float yDiff   = std::abs(a.rect.center.y - b.rect.center.y) / std::max(aLen, bLen);

    return dirDiff < p.maxAngleDiff && lenDiff < p.maxLenDiff && yDiff < p.maxYDiff;
}

bool matchLightBars(const std::vector<LightBar>& bars, const Params& p,
                    LightBar& left, LightBar& right)
{
    bool found = false;
    float bestScore = -1.f;
    for (size_t i = 0; i < bars.size(); ++i)
    {
        for (size_t j = i + 1; j < bars.size(); ++j)
        {
            if (!isPair(bars[i], bars[j], p)) continue;

            float score = std::min(bars[i].length, bars[j].length);
            if (score <= bestScore) continue;

            bestScore = score;
            bool iIsLeft = (bars[i].rect.center.x < bars[j].rect.center.x);
            left  = iIsLeft ? bars[i] : bars[j];
            right = iIsLeft ? bars[j] : bars[i];
            found = true;
        }
    }
    return found;
}

void drawLightBar(cv::Mat& img, const LightBar& bar, const cv::Scalar& color, int thickness = 1)
{
    cv::Point2f pts[4];
    bar.rect.points(pts);
    for (int i = 0; i < 4; ++i)
        cv::line(img, pts[i], pts[(i + 1) % 4], color, thickness);
}

void drawArmor(cv::Mat& img, const std::vector<cv::Point2f>& corners)
{
    for (int i = 0; i < 4; ++i)
        cv::line(img, corners[i], corners[(i + 1) % 4], cv::Scalar(0, 255, 0), 2);
    for (const auto& pt : corners)
        cv::circle(img, pt, 3, cv::Scalar(0, 255, 0), -1);
}

cv::Mat detectAndDraw(const cv::Mat& frame, const Params& p, const Options& opt)
{
    cv::Mat result = frame.clone();

    cv::Mat mask = makeMask(frame, p, opt);
    std::vector<LightBar> bars = findLightBars(mask, p);

    if (opt.debug)
        for (const auto& b : bars)
            drawLightBar(result, b, cv::Scalar(0, 0, 255), 1);

    LightBar left, right;
    if (!matchLightBars(bars, p, left, right))
        return result;

    std::vector<cv::Point2f> corners = { left.top, right.top, right.bottom, left.bottom };

    drawLightBar(result, left,  cv::Scalar(0, 255, 255), 1);
    drawLightBar(result, right, cv::Scalar(0, 255, 255), 1);
    drawArmor(result, corners);

    return result;
}

int main(int argc, char** argv)
{
    if (argc < 2)
    {
        std::cout << "Usage: " << argv[0]
                  << " <video> [red|blue] [--hsv] [--mask] [--debug] [--save]\n"
                  << "Example: " << argv[0] << " video.avi blue --mask" << std::endl;
        return -1;
    }

    std::string videoPath = argv[1];
    Options opt;
    for (int i = 2; i < argc; ++i)
    {
        std::string a = argv[i];
        if      (a == "red")     opt.color = "red";
        else if (a == "blue")    opt.color = "blue";
        else if (a == "--hsv")   opt.useHsv = true;
        else if (a == "--mask")  opt.showMask = true;
        else if (a == "--debug") opt.debug = true;
        else if (a == "--save")  opt.saveVideo = true;
    }
    Params p;

    cv::VideoCapture cap(videoPath);
    if (!cap.isOpened())
    {
        std::cerr << "无法打开视频文件: " << videoPath << std::endl;
        return -1;
    }

    cv::VideoWriter writer;
    if (opt.saveVideo)
    {
        int fps = static_cast<int>(cap.get(cv::CAP_PROP_FPS));
        if (fps <= 0) fps = 30;
        int w = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
        int h = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
        writer.open("output.avi", cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), fps, cv::Size(w, h));
        if (!writer.isOpened())
            std::cerr << "警告: 无法创建 输出.avi,将只处理不保存" << std::endl;
    }

    bool guiOk = true;
    cv::Mat frame;
    while (true)
    {
        cap >> frame;
        if (frame.empty()) break;

        cv::Mat result = detectAndDraw(frame, p, opt);

        if (guiOk)
        {
            try
            {
                cv::imshow("light_bar", result);
                if (opt.showMask)
                    cv::imshow("mask", makeMask(frame, p, opt));
                if (cv::waitKey(20) == 27) break;
            }
            catch (const cv::Exception&)
            {
                guiOk = false;
                std::cerr << "没有可用的显示; 在没有窗口的情况下处理" << std::endl;
            }
        }

        if (writer.isOpened())
            writer.write(result);
    }

    writer.release();
    std::cout << "处理完成" << (opt.saveVideo ? " - saved to output.avi" : "") << std::endl;
    return 0;
}
