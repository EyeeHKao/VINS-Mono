#pragma once

#include <cstdio>
#include <iostream>
#include <queue>
#include <execinfo.h>
#include <csignal>

#include <opencv2/opencv.hpp>
#include <eigen3/Eigen/Dense>

#include "camodocal/camera_models/CameraFactory.h"
#include "camodocal/camera_models/CataCamera.h"
#include "camodocal/camera_models/PinholeCamera.h"

#include "parameters.h"
#include "tic_toc.h"

using namespace std;
using namespace camodocal;
using namespace Eigen;

bool inBorder(const cv::Point2f &pt);

void reduceVector(vector<cv::Point2f> &v, vector<uchar> status);
void reduceVector(vector<int> &v, vector<uchar> status);

class FeatureTracker
{
  public:
    FeatureTracker();

    void readImage(const cv::Mat &_img,double _cur_time);

    void setMask();

    void addPoints();

    bool updateID(unsigned int i);

    void readIntrinsicParameter(const string &calib_file);

    void showUndistortion(const string &name);

    void rejectWithF();

    void undistortedPoints();

    cv::Mat mask;   ///图像mask,用于取感兴趣部分的图像
    cv::Mat fisheye_mask; ///鱼眼形的mask
    cv::Mat prev_img, cur_img, forw_img; ///前一帧， 当前帧， 下一帧图像
    vector<cv::Point2f> n_pts;  ///特征点集
    vector<cv::Point2f> prev_pts, cur_pts, forw_pts;  ///前一帧， 当前帧， 下一帧特征点集
    vector<cv::Point2f> prev_un_pts, cur_un_pts; ///前一帧， 当前帧 未追踪/匹配的特帧点集
    vector<cv::Point2f> pts_velocity; 点速度
    vector<int> ids; ///特征点匹配关系
    vector<int> track_cnt;
    map<int, cv::Point2f> cur_un_pts_map;
    map<int, cv::Point2f> prev_un_pts_map;
    camodocal::CameraPtr m_camera;  ///相机模型
    double cur_time;
    double prev_time;

    static int n_id;
};
