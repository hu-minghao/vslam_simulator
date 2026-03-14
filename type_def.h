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
    Eigen::Vector3d p;
};

struct Observation
{
    int pose_id;
    int landmark_id;
    Eigen::Vector2d uv;
};