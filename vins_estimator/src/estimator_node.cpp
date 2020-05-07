#include <stdio.h>
#include <queue>
#include <map>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>

#include "estimator.h"
#include "parameters.h"
#include "utility/visualization.h"


Estimator estimator;

std::condition_variable con;
double current_time = -1;
queue<sensor_msgs::ImuConstPtr> imu_buf;
queue<sensor_msgs::PointCloudConstPtr> feature_buf;
queue<sensor_msgs::PointCloudConstPtr> relo_buf;
int sum_of_wait = 0;

std::mutex m_buf;   ///三个缓冲队列对应的互斥量
std::mutex m_state; ///估计器m_estimator的状态的互斥量，
std::mutex i_buf;   ///没用到
std::mutex m_estimator; ///调用估计器成员函数的互斥量，主要是在处理IMU和image时

//imu向前传播时的临时存储状态
double latest_time; ///最新的imu时间戳
Eigen::Vector3d tmp_P;  
Eigen::Quaterniond tmp_Q;
Eigen::Vector3d tmp_V;
Eigen::Vector3d tmp_Ba;
Eigen::Vector3d tmp_Bg;

//上一时刻的imu测量值
Eigen::Vector3d acc_0;  
Eigen::Vector3d gyr_0;

bool init_feature = 0;
bool init_imu = 1;
double last_imu_t = 0;

//imu预测，根据tmp_Q/V/Ba/Bg向前传播，结果仍保存在这些变量中
void predict(const sensor_msgs::ImuConstPtr &imu_msg)
{
    double t = imu_msg->header.stamp.toSec();
    if (init_imu)
    {
        latest_time = t;
        init_imu = 0;
        return;
    }
    double dt = t - latest_time;
    latest_time = t;

    double dx = imu_msg->linear_acceleration.x;
    double dy = imu_msg->linear_acceleration.y;
    double dz = imu_msg->linear_acceleration.z;
    Eigen::Vector3d linear_acceleration{dx, dy, dz};

    double rx = imu_msg->angular_velocity.x;
    double ry = imu_msg->angular_velocity.y;
    double rz = imu_msg->angular_velocity.z;
    Eigen::Vector3d angular_velocity{rx, ry, rz};

    //中点积分，非欧拉积分
    Eigen::Vector3d un_acc_0 = tmp_Q * (acc_0 - tmp_Ba) - estimator.g;

    Eigen::Vector3d un_gyr = 0.5 * (gyr_0 + angular_velocity) - tmp_Bg;
    tmp_Q = tmp_Q * Utility::deltaQ(un_gyr * dt);

    Eigen::Vector3d un_acc_1 = tmp_Q * (linear_acceleration - tmp_Ba) - estimator.g;

    Eigen::Vector3d un_acc = 0.5 * (un_acc_0 + un_acc_1);

    tmp_P = tmp_P + dt * tmp_V + 0.5 * dt * dt * un_acc;
    tmp_V = tmp_V + dt * un_acc;

    acc_0 = linear_acceleration;
    gyr_0 = angular_velocity;
}

//根据窗口中最新的估计的状态（处理完图像后，imu_buf里又会产生一些imu测量值），向前传播imu预测值
void update()
{
    TicToc t_predict;
    latest_time = current_time;
    tmp_P = estimator.Ps[WINDOW_SIZE];
    tmp_Q = estimator.Rs[WINDOW_SIZE];
    tmp_V = estimator.Vs[WINDOW_SIZE];
    tmp_Ba = estimator.Bas[WINDOW_SIZE];
    tmp_Bg = estimator.Bgs[WINDOW_SIZE];
    acc_0 = estimator.acc_0;
    gyr_0 = estimator.gyr_0;

    queue<sensor_msgs::ImuConstPtr> tmp_imu_buf = imu_buf;
    for (sensor_msgs::ImuConstPtr tmp_imu_msg; !tmp_imu_buf.empty(); tmp_imu_buf.pop())
        predict(tmp_imu_buf.front());

}

/*
获取缓存在全局变量imu_buf和featrue_buf中的IMU数据和图像帧数据，直到其中一个队列为空，并把数据做一个简单的对齐
*/

std::vector<std::pair<std::vector<sensor_msgs::ImuConstPtr>, sensor_msgs::PointCloudConstPtr>>
getMeasurements()
{
    std::vector<std::pair<std::vector<sensor_msgs::ImuConstPtr>, sensor_msgs::PointCloudConstPtr>> measurements;

    while (true)
    {
        //其中一个队列为空时返回
        if (imu_buf.empty() || feature_buf.empty())
            return measurements;
        //IMU最新数据的时间戳小于等于图像帧最旧的数据的时间戳，说明IMU数据过时了，需要等待后续IMU数据
        if (!(imu_buf.back()->header.stamp.toSec() > feature_buf.front()->header.stamp.toSec() + estimator.td))
        {
            //可能是在系统刚启动时先采集的IMU，后采集的图像
            //ROS_WARN("wait for imu, only should happen at the beginning") 
            sum_of_wait++;
            return measurements;
        }
        //IMU最旧的数据大于等于图像帧最旧的数据，则抛弃这帧图像，说明这帧图像过时了，需要抛弃
        if (!(imu_buf.front()->header.stamp.toSec() < feature_buf.front()->header.stamp.toSec() + estimator.td))
        {
            //可是是在系统刚启动时先采集了图像，后采集了IMU
            ROS_WARN("throw img, only should happen at the beginning");
            feature_buf.pop();
            continue;
        }
        sensor_msgs::PointCloudConstPtr img_msg = feature_buf.front();
        feature_buf.pop();

        std::vector<sensor_msgs::ImuConstPtr> IMUs;
        while (imu_buf.front()->header.stamp.toSec() < img_msg->header.stamp.toSec() + estimator.td)
        {
            IMUs.emplace_back(imu_buf.front());
            imu_buf.pop();
        }
        //补充再加入一个紧随image数据后的IMU数据，用于后续加权求出和这个image数据“对齐”了的“假”IMU数据
        //注意，这个数据并不会弹出，采取下一组数据时会继续用到的
        IMUs.emplace_back(imu_buf.front());
        if (IMUs.empty())
            ROS_WARN("no imu between two image");
        measurements.emplace_back(IMUs, img_msg);
    }
    return measurements;
}

void imu_callback(const sensor_msgs::ImuConstPtr &imu_msg)
{
    if (imu_msg->header.stamp.toSec() <= last_imu_t)
    {
        ROS_WARN("imu message in disorder!");
        return;
    }
    last_imu_t = imu_msg->header.stamp.toSec();

    m_buf.lock();
    imu_buf.push(imu_msg);
    m_buf.unlock();
    con.notify_one();

    last_imu_t = imu_msg->header.stamp.toSec();

    {
        std::lock_guard<std::mutex> lg(m_state);
        predict(imu_msg);
        std_msgs::Header header = imu_msg->header;
        header.frame_id = "world";
        if (estimator.solver_flag == Estimator::SolverFlag::NON_LINEAR)
            pubLatestOdometry(tmp_P, tmp_Q, tmp_V, header);
    }
}


void feature_callback(const sensor_msgs::PointCloudConstPtr &feature_msg)
{
    if (!init_feature)
    {
        //skip the first detected feature, which doesn't contain optical flow speed
        init_feature = 1;
        return;
    }
    m_buf.lock();
    feature_buf.push(feature_msg);
    m_buf.unlock();
    con.notify_one();
}

void restart_callback(const std_msgs::BoolConstPtr &restart_msg)
{
    if (restart_msg->data == true)
    {
        ROS_WARN("restart the estimator!");
        m_buf.lock();
        while(!feature_buf.empty())
            feature_buf.pop();
        while(!imu_buf.empty())
            imu_buf.pop();
        m_buf.unlock();
        m_estimator.lock();
        estimator.clearState();
        estimator.setParameter();
        m_estimator.unlock();
        current_time = -1;
        last_imu_t = 0;
    }
    return;
}
//重定位回调，将重定位帧或者回环帧，放入缓存队列：
void relocalization_callback(const sensor_msgs::PointCloudConstPtr &points_msg)
{
    //printf("relocalization callback! \n");
    m_buf.lock();
    relo_buf.push(points_msg);
    m_buf.unlock();
}

// thread: visual-inertial odometry
void process()
{
    while (true)
    {
        std::vector<std::pair<std::vector<sensor_msgs::ImuConstPtr>, sensor_msgs::PointCloudConstPtr>> measurements;
        //要从imu_buf和feature_buf取数据，先锁住缓冲区：
        std::unique_lock<std::mutex> lk(m_buf);
        //使用条件变量，等待缓冲区有数据，并对齐后取出：
        con.wait(lk, [&]
                 {
            return (measurements = getMeasurements()).size() != 0;
                 });
        //解锁
        lk.unlock();
        //下面进入估计位姿过程，需要使用Estimator的相关成员函数，所以先上锁锁住，防止其他线程也调用了Estimator的相关成员函数：
        m_estimator.lock();
        for (auto &measurement : measurements)
        {
            //每个measurement由一组IMU数据和紧随其后的Image数据组成，而且，根据数据对齐的方式，最后一个IMU数据肯定是等于或者恰好大于image数据的时间戳的
            //这么做是要是通过夹在image数据前后两个IMU数据基于时间加权获得image时间戳对应“假”IMU数据，相当于IMU数据和image数据做了对齐
            auto img_msg = measurement.second;
            double dx = 0, dy = 0, dz = 0, rx = 0, ry = 0, rz = 0;
            for (auto &imu_msg : measurement.first)
            {
                double t = imu_msg->header.stamp.toSec();
                double img_t = img_msg->header.stamp.toSec() + estimator.td;
                //在这里处理image数据之前的几帧IMU数据
                if (t <= img_t)
                { 
                    //第一次处理时，dt=0
                    if (current_time < 0)
                        current_time = t;
                    double dt = t - current_time;
                    ROS_ASSERT(dt >= 0);
                    current_time = t;
                    dx = imu_msg->linear_acceleration.x;
                    dy = imu_msg->linear_acceleration.y;
                    dz = imu_msg->linear_acceleration.z;
                    rx = imu_msg->angular_velocity.x;
                    ry = imu_msg->angular_velocity.y;
                    rz = imu_msg->angular_velocity.z;
                    //计算IMU预计分值，并向前传播imu测量量，
                    estimator.processIMU(dt, Vector3d(dx, dy, dz), Vector3d(rx, ry, rz));
                    //printf("imu: dt:%f a: %f %f %f w: %f %f %f\n",dt, dx, dy, dz, rx, ry, rz);

                }
                //在这里处理image数据后的那帧IMU数据，
                else
                {
                    double dt_1 = img_t - current_time;
                    double dt_2 = t - img_t;
                    //当前时间重新确定为imgage数据对应的时间戳,而不是用其后一帧IMU数据的时间戳t赋值：
                    current_time = img_t;
                    ROS_ASSERT(dt_1 >= 0);
                    ROS_ASSERT(dt_2 >= 0);
                    ROS_ASSERT(dt_1 + dt_2 > 0);
                    double w1 = dt_2 / (dt_1 + dt_2);
                    double w2 = dt_1 / (dt_1 + dt_2);
                    //基于时间进行加权对齐得到image数据时间戳下的“假”IMU数据
                    dx = w1 * dx + w2 * imu_msg->linear_acceleration.x;
                    dy = w1 * dy + w2 * imu_msg->linear_acceleration.y;
                    dz = w1 * dz + w2 * imu_msg->linear_acceleration.z;
                    rx = w1 * rx + w2 * imu_msg->angular_velocity.x;
                    ry = w1 * ry + w2 * imu_msg->angular_velocity.y;
                    rz = w1 * rz + w2 * imu_msg->angular_velocity.z;
                    estimator.processIMU(dt_1, Vector3d(dx, dy, dz), Vector3d(rx, ry, rz));
                    //printf("dimu: dt:%f a: %f %f %f w: %f %f %f\n",dt_1, dx, dy, dz, rx, ry, rz);
                }
            }
            // set relocalization frame
            sensor_msgs::PointCloudConstPtr relo_msg = NULL;
            //在处理下一帧image前，检测重定位帧（回环）消息缓存队列：重定位帧是指与过去关键帧发生了回环的窗口中的某一帧
            //注意：重定位帧的数据存储方式和图像特征点帧(feature_track发布出来的)不同：
            //取最新的重定位帧并清空缓存队列
            while (!relo_buf.empty())
            {
                relo_msg = relo_buf.front();
                relo_buf.pop();
            }
            if (relo_msg != NULL)
            {
                vector<Vector3d> match_points;
                //重定位帧的时间戳
                double frame_stamp = relo_msg->header.stamp.toSec();
                for (unsigned int i = 0; i < relo_msg->points.size(); i++)
                {
                    Vector3d u_v_id;
                    //这三个域分别是归一化坐标系下x,y 以及特征点ID，注意区分：
                    u_v_id.x() = relo_msg->points[i].x;
                    u_v_id.y() = relo_msg->points[i].y;
                    u_v_id.z() = relo_msg->points[i].z;
                    //将回环帧中的特征点的相机归一化坐标和ID放入匹配点集中
                    match_points.push_back(u_v_id);
                }
                //重定位帧的位姿（回环检测四自由度优化计算出的）：这个域下分别代表了该帧的位置和姿态四元数：
                Vector3d relo_t(relo_msg->channels[0].values[0], relo_msg->channels[0].values[1], relo_msg->channels[0].values[2]);
                Quaterniond relo_q(relo_msg->channels[0].values[3], relo_msg->channels[0].values[4], relo_msg->channels[0].values[5], relo_msg->channels[0].values[6]);
                Matrix3d relo_r = relo_q.toRotationMatrix();
                int frame_index;
                //重定位帧的帧号
                frame_index = relo_msg->channels[0].values[7];
                //设置重定位帧的相关信息：位姿，匹配点信息
                estimator.setReloFrame(frame_stamp, frame_index, match_points, relo_t, relo_r);
            }
            //开始处理这帧图像数据了：
            ROS_DEBUG("processing vision data with stamp %f \n", img_msg->header.stamp.toSec());
            
            TicToc t_s;
            map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> image;
            //将img_msg转换为image的格式：
            for (unsigned int i = 0; i < img_msg->points.size(); i++)
            {
                int v = img_msg->channels[0].values[i] + 0.5;
                //特征点ID：
                int feature_id = v / NUM_OF_CAM;
                //相机ID：单目就只有一个相机：
                int camera_id = v % NUM_OF_CAM;
                //特征点在相机归一化坐标系下的坐标：
                double x = img_msg->points[i].x;
                double y = img_msg->points[i].y;
                double z = img_msg->points[i].z;
                //该帧下的像素坐标：
                double p_u = img_msg->channels[1].values[i];
                double p_v = img_msg->channels[2].values[i];
                //像素速度或光流：
                double velocity_x = img_msg->channels[3].values[i];
                double velocity_y = img_msg->channels[4].values[i];
                ROS_ASSERT(z == 1);
                Eigen::Matrix<double, 7, 1> xyz_uv_velocity;
                xyz_uv_velocity << x, y, z, p_u, p_v, velocity_x, velocity_y;
                //构造image
                image[feature_id].emplace_back(camera_id,  xyz_uv_velocity);
            }
            //处理图像数据：从featrue_track传出来的是ROS消息类型sensor_msgs::PointCloudConstPtr,这里给转化为image的类型
            //这样做虽然麻烦，但是在不用ROS时，就显得很实用了
            estimator.processImage(image, img_msg->header);

            double whole_t = t_s.toc();
            printStatistics(estimator, whole_t);
            std_msgs::Header header = img_msg->header;
            header.frame_id = "world";
            //发布消息： 
            pubOdometry(estimator, header); ///发布窗口中最新帧对应的imu系的位姿及路径在漂移的世界坐标系下和校正的世界坐标系下
            pubKeyPoses(estimator, header); ///发布窗口中所有imu的位置
            pubCameraPose(estimator, header);///发布最新的相机位姿
            pubPointCloud(estimator, header);///发布点云及边缘化最旧帧的点云
            pubTF(estimator, header);  ///发布tf坐标变换
            pubKeyframe(estimator); ///发布次次新关键帧，目前窗口中的WINDOW_SIZE-2帧
            if (relo_msg != NULL)
                pubRelocalization(estimator);   ///如果窗口中有重定位帧，发布重定位帧和回环帧的相对位姿
            //ROS_ERROR("end: %f, at %f", img_msg->header.stamp.toSec(), ros::Time::now().toSec());
        }
        //到这里才解锁啊：主要是方便省事，不然上面那个循环里面要不停的加锁上锁：但这会不会导致一些问题，
        //因为每次都是处理好几组IMUS-image数据对，有时组数多，有时组数少，组数多的时候可能处理的比较慢，这样就会影响锁住的时间长短，导致好几组过后才输出，这个要在平台上实测
        m_estimator.unlock();
        //缓冲区再次锁住，取此时imu_buf
        m_buf.lock();
        //估计器状态上锁，因为要判断是在初始化还是在非线性优化：
        m_state.lock();
        //初始化状态，说明还没完成初始化，不能更新：
        if (estimator.solver_flag == Estimator::SolverFlag::NON_LINEAR)
            //更新至最新状态，将最新的也即窗口中最后一帧的估计出的位姿，速度，IMU随机游走偏差等，通过最新缓存的imu数据更新到全局变量中tmp_P/Q...：
            //可见，这里在时间上没有均匀的以imu的频率间更新状态（或者说没有及时处理imu数据，而是缓存一部分后批量处理，不实时）
            update();   
        m_state.unlock();
        m_buf.unlock();
    }
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "vins_estimator");
    ros::NodeHandle n("~");
    ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Info);
    readParameters(n);
    //设置相机IMU间的外参数,和延时，可通过事先标定获取:
    estimator.setParameter();
#ifdef EIGEN_DONT_PARALLELIZE
    ROS_DEBUG("EIGEN_DONT_PARALLELIZE");
#endif
    ROS_WARN("waiting for image and imu...");

    registerPub(n);

    ros::Subscriber sub_imu = n.subscribe(IMU_TOPIC, 2000, imu_callback, ros::TransportHints().tcpNoDelay());
    ros::Subscriber sub_image = n.subscribe("/feature_tracker/feature", 2000, feature_callback);
    ros::Subscriber sub_restart = n.subscribe("/feature_tracker/restart", 2000, restart_callback);
    ros::Subscriber sub_relo_points = n.subscribe("/pose_graph/match_points", 2000, relocalization_callback);

    std::thread measurement_process{process};
    ros::spin();

    return 0;
}
