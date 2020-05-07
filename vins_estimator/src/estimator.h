#pragma once

#include "parameters.h"
#include "feature_manager.h"
#include "utility/utility.h"
#include "utility/tic_toc.h"
#include "initial/solve_5pts.h"
#include "initial/initial_sfm.h"
#include "initial/initial_alignment.h"
#include "initial/initial_ex_rotation.h"
#include <std_msgs/Header.h>
#include <std_msgs/Float32.h>

#include <ceres/ceres.h>
#include "factor/imu_factor.h"
#include "factor/pose_local_parameterization.h"
#include "factor/projection_factor.h"
#include "factor/projection_td_factor.h"
#include "factor/marginalization_factor.h"

#include <unordered_map>
#include <queue>
#include <opencv2/core/eigen.hpp>


class Estimator
{
  public:
    Estimator();

    void setParameter();

    // interface
    void processIMU(double t, const Vector3d &linear_acceleration, const Vector3d &angular_velocity);
    void processImage(const map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> &image, const std_msgs::Header &header);
    void setReloFrame(double _frame_stamp, int _frame_index, vector<Vector3d> &_match_points, Vector3d _relo_t, Matrix3d _relo_r);

    // internal
    void clearState();
    bool initialStructure();
    bool visualInitialAlign();
    bool relativePose(Matrix3d &relative_R, Vector3d &relative_T, int &l);
    void slideWindow();
    void solveOdometry();
    void slideWindowNew();
    void slideWindowOld();
    void optimization();
    void vector2double();
    void double2vector();
    bool failureDetection();


    enum SolverFlag
    {
        INITIAL,
        NON_LINEAR
    };

    
    enum MarginalizationFlag
    {
        MARGIN_OLD = 0,　　　　///为０表示边缘化窗口中最旧帧（当新来的帧为关键帧时）
        MARGIN_SECOND_NEW = 1 ///为１表示边缘化窗口的次新帧（当新来的帧不是关键帧时）
    };

    SolverFlag solver_flag;
    MarginalizationFlag  marginalization_flag;
    Vector3d g;
    MatrixXd Ap[2], backup_A;
    VectorXd bp[2], backup_b;

    Matrix3d ric[NUM_OF_CAM];
    Vector3d tic[NUM_OF_CAM];

    //下面数组中加１都表示还要包含最新来的帧，即当前帧或者说是最新帧
    //那么,当前帧/或最新帧的索引就是WINDOW_SIZE
    //窗口中次新帧的索引就是WINDOW_SIZE-1
    Vector3d Ps[(WINDOW_SIZE + 1)]; 　///窗口中所有帧对应imu位置＋新来的帧对应的imu的位置
    Vector3d Vs[(WINDOW_SIZE + 1)]; 　///速度
    Matrix3d Rs[(WINDOW_SIZE + 1)];　　///姿态
    Vector3d Bas[(WINDOW_SIZE + 1)];  ///加速度偏差
    Vector3d Bgs[(WINDOW_SIZE + 1)];　///陀螺仪偏差
    double td;  ///imu数据和image数据的时间延迟

    Matrix3d back_R0, last_R, last_R0;
    Vector3d back_P0, last_P, last_P0;
    std_msgs::Header Headers[(WINDOW_SIZE + 1)];  ///时间戳，序号等header信息

    IntegrationBase *pre_integrations[(WINDOW_SIZE + 1)]; ///窗口中的预积分信息，假设下标为i, 那么表示i-1(上一帧，可能不在窗口中了)帧到i帧的预积分，所以总共WINDOW_SIZE+1个
    Vector3d acc_0, gyr_0;  ///

    vector<double> dt_buf[(WINDOW_SIZE + 1)];　///窗口中的上一帧i-1到当前帧i的之间所有imu数据的时间间隔的数组，
    vector<Vector3d> linear_acceleration_buf[(WINDOW_SIZE + 1)];  ///窗口中的上一帧i-1到当前帧i的之间所有imu加速度数据的数组
    vector<Vector3d> angular_velocity_buf[(WINDOW_SIZE + 1)];///窗口中的上一帧i-1到当前帧i的之间所有imu角速度数据的数组

    int frame_count;
    int sum_of_outlier, sum_of_back, sum_of_front, sum_of_invalid;

    FeatureManager f_manager;
    MotionEstimator m_estimator;
    InitialEXRotation initial_ex_rotation;

    bool first_imu;
    bool is_valid, is_key;
    bool failure_occur;

    vector<Vector3d> point_cloud;
    vector<Vector3d> margin_cloud;
    vector<Vector3d> key_poses; ///
    double initial_timestamp;


    double para_Pose[WINDOW_SIZE + 1][SIZE_POSE]; ///位置姿态放一起，
    double para_SpeedBias[WINDOW_SIZE + 1][SIZE_SPEEDBIAS]; ///速度偏差放一起
    double para_Feature[NUM_OF_F][SIZE_FEATURE];  ///
    double para_Ex_Pose[NUM_OF_CAM][SIZE_POSE]; ///相机外参，多个相机多个外参
    double para_Retrive_Pose[SIZE_POSE];  ///恢复的位姿
    double para_Td[1][1];
    double para_Tr[1][1];

    int loop_window_index;

    MarginalizationInfo *last_marginalization_info;
    vector<double *> last_marginalization_parameter_blocks;

    map<double, ImageFrame> all_image_frame;
    IntegrationBase *tmp_pre_integration;

    //relocalization variable
    bool relocalization_info;
    double relo_frame_stamp;  ///窗口中重定位帧的时间戳
    double relo_frame_index;  ///重定位帧在窗口中的索引
    int relo_frame_local_index;　///?
    vector<Vector3d> match_points;  ///重定位帧对应的匹配上的3d点
    double relo_Pose[SIZE_POSE];  ///重定位帧的位姿（回环ba优化后的）
    Matrix3d drift_correct_r; ///漂移world系到校正的world系的姿态
    Vector3d drift_correct_t;　///.....平移
    Vector3d prev_relo_t;　///回环帧在漂移系下的位移和姿态
    Matrix3d prev_relo_r;
    Vector3d relo_relative_t; ///重定位帧和回环帧的相对位姿和相对yaw角
    Quaterniond relo_relative_q;
    double relo_relative_yaw;
};
