#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <vector>
#include <random>
#include "type_def.h"

class Simulator
{
public:

    std::vector<Pose> poses;
    std::vector<Landmark> landmarks;
    std::vector<Observation> observations;

    double fx = 500;
    double fy = 500;
    double cx = 320;
    double cy = 240;

    int width = 640;
    int height = 480;

    void generateTrajectory(int N)
    {
        poses.clear();

        for(int i=0;i<N;i++)
        {
            double theta = i * 2.0 * M_PI / N;

            Eigen::Vector3d t;
            t << 8*cos(theta), 8*sin(theta), 3;

            Eigen::Matrix3d R =
                Eigen::AngleAxisd(-theta, Eigen::Vector3d::UnitZ()).toRotationMatrix();

            Pose p;
            p.R = R;
            p.t = t;

            poses.push_back(p);
        }
    }

    void generateLandmarks(int N)
    {
        landmarks.clear();

        std::default_random_engine gen;
        std::uniform_real_distribution<double> dist(-10,10);

        for(int i=0;i<N;i++)
        {
            Landmark lm;
            lm.id = i;

            lm.p << dist(gen), dist(gen), dist(gen)+8;

            landmarks.push_back(lm);
        }
    }

    bool project(const Eigen::Vector3d &Pc, Eigen::Vector2d &uv)
    {
        if(Pc.z() <= 0)
            return false;

        double u = fx * Pc.x()/Pc.z() + cx;
        double v = fy * Pc.y()/Pc.z() + cy;

        if(u<0 || u>=width) return false;
        if(v<0 || v>=height) return false;

        uv << u,v;
        return true;
    }

    void generateObservations()
    {
        observations.clear();

        std::default_random_engine gen;
        std::normal_distribution<double> noise(0,1.0);

        for(int i=0;i<poses.size();i++)
        {
            const Pose &pose = poses[i];

            for(auto &lm:landmarks)
            {
                Eigen::Vector3d Pc =
                    pose.R.transpose()*(lm.p - pose.t);

                Eigen::Vector2d uv;

                if(project(Pc,uv))
                {
                    uv.x() += noise(gen);
                    uv.y() += noise(gen);

                    Observation obs;

                    obs.pose_id = i;
                    obs.landmark_id = lm.id;
                    obs.uv = uv;

                    observations.push_back(obs);
                }
            }
        }
    }

    void generate(int pose_num,int landmark_num)
    {
        generateTrajectory(pose_num);
        generateLandmarks(landmark_num);
        generateObservations();
    }
};