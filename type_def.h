#pragma once
#include <Eigen/Core>

struct Pose
{
    Eigen::Matrix3d R;
    Eigen::Vector3d t;
};

struct Landmark
{
    int id;

    Eigen::Vector3d p_gt;
    Eigen::Vector3d p_noisy;
};

struct Observation
{
    int pose_id;
    int landmark_id;
    int feature_id;

    Eigen::Vector2d uv_gt;
    Eigen::Vector2d uv_noisy;

    bool is_outlier=false;
};

struct FeatureObservation
{
    int landmark_id;
    int feature_id;

    Eigen::Vector2d uv;
};

struct Frame
{
    int id;

    double timestamp;

    Pose pose_gt;
    Pose pose_noisy;

    std::vector<FeatureObservation> features;
};