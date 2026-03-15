#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <vector>
#include <random>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <sys/stat.h>
#include "type_def.h"
using namespace std;
class Simulator
{
public:
    vector<Pose> poses_gt;
    vector<Pose> poses_noisy;

    vector<Landmark> landmarks;

    vector<Observation> observations;

    // 按帧索引观测（高效）
    vector<vector<Observation>> obs_per_frame;

    double fx = 500;
    double fy = 500;
    double cx = 320;
    double cy = 240;

    int width = 640;
    int height = 480;

    double pixel_noise_sigma = 1.0;
    double pose_noise_sigma = 0.05;
    double landmark_noise_sigma = 0.1;

    double outlier_ratio = 0.05;
    double dropout_ratio = 0.05;

    double frame_dt = 0.1;

    int current_frame_index = 0;

    default_random_engine gen;

public:
    void generateTrajectory(int N)
    {
        poses_gt.clear();

        for (int i = 0; i < N; i++)
        {
            double theta = i * 2.0 * M_PI / N;

            Pose p;

            p.t << 8 * cos(theta), 8 * sin(theta), 3;

            p.R =
                Eigen::AngleAxisd(-theta, Eigen::Vector3d::UnitZ()).toRotationMatrix();

            poses_gt.push_back(p);
        }
    }

    void addPoseNoise()
    {
        poses_noisy = poses_gt;

        normal_distribution<double> noise(0, pose_noise_sigma);

        for (auto &p : poses_noisy)
        {
            p.t += Eigen::Vector3d(
                noise(gen),
                noise(gen),
                noise(gen));
        }
    }

    void generateLandmarks(int N)
    {
        landmarks.clear();

        uniform_real_distribution<double> dist(-10, 10);

        for (int i = 0; i < N; i++)
        {
            Landmark lm;

            lm.id = i;

            lm.p_gt << dist(gen), dist(gen), dist(gen) + 8;

            landmarks.push_back(lm);
        }
    }

    void addLandmarkNoise()
    {
        normal_distribution<double> noise(0, landmark_noise_sigma);

        for (auto &lm : landmarks)
        {
            lm.p_noisy =
                lm.p_gt +
                Eigen::Vector3d(
                    noise(gen),
                    noise(gen),
                    noise(gen));
        }
    }

    bool project(const Eigen::Vector3d &Pc, Eigen::Vector2d &uv)
    {
        if (Pc.z() <= 0)
            return false;

        double u = fx * Pc.x() / Pc.z() + cx;
        double v = fy * Pc.y() / Pc.z() + cy;

        if (u < 0 || u >= width)
            return false;
        if (v < 0 || v >= height)
            return false;

        uv << u, v;

        return true;
    }

    void generateObservations()
    {
        observations.clear();

        obs_per_frame.clear();

        obs_per_frame.resize(poses_gt.size());

        normal_distribution<double> noise(0, pixel_noise_sigma);
        uniform_real_distribution<double> prob(0, 1);

        for (int i = 0; i < poses_gt.size(); i++)
        {
            int feature_index = 0;

            for (auto &lm : landmarks)
            {
                Eigen::Vector3d Pc =
                    poses_gt[i].R.transpose() *
                    (lm.p_gt - poses_gt[i].t);

                Eigen::Vector2d uv;

                if (!project(Pc, uv))
                    continue;

                if (prob(gen) < dropout_ratio)
                    continue;

                Observation obs;

                obs.pose_id = i;

                obs.landmark_id = lm.id;

                obs.feature_id = feature_index++;

                obs.uv_gt = uv;

                obs.uv_noisy =
                    uv +
                    Eigen::Vector2d(
                        noise(gen),
                        noise(gen));

                if (prob(gen) < outlier_ratio)
                {
                    obs.is_outlier = true;

                    uniform_real_distribution<double> out_u(0, width);
                    uniform_real_distribution<double> out_v(0, height);

                    obs.uv_noisy << out_u(gen),
                        out_v(gen);
                }

                observations.push_back(obs);

                obs_per_frame[i].push_back(observations.back());
            }
        }
    }

    bool getNextFrame(Frame &frame, bool use_noise = true)
    {
        if (current_frame_index >= poses_gt.size())
            return false;

        int i = current_frame_index;

        frame.id = i;

        frame.timestamp = i * frame_dt;

        frame.pose_gt = poses_gt[i];

        frame.pose_noisy = poses_noisy[i];

        frame.features.clear();

        for (auto obs : obs_per_frame[i])
        {
            FeatureObservation fo;
            fo.landmark_id = obs.landmark_id;
            fo.feature_id = obs.feature_id;

            fo.uv =
                use_noise ? obs.uv_noisy : obs.uv_gt;
            frame.features.push_back(fo);
        }
        current_frame_index++;

        return true;
    }

    void reset()
    {
        current_frame_index = 0;
    }

    void saveFrames(string folder, double dt, bool use_noise)
    {
        mkdir(folder.c_str(), 0777);

        for (int i = 0; i < poses_gt.size(); i++)
        {
            double timestamp = i * dt;

            stringstream ss;

            ss << folder << "/"
               << fixed << setprecision(3)
               << timestamp << ".txt";

            ofstream fout(ss.str());

            fout << i << endl;

            Eigen::Quaterniond q(poses_gt[i].R);

            fout << poses_gt[i].t.x() << " "
                 << poses_gt[i].t.y() << " "
                 << poses_gt[i].t.z() << " "
                 << q.x() << " "
                 << q.y() << " "
                 << q.z() << " "
                 << q.w() << endl;

            for (auto &obs : observations)
            {
                if (obs.pose_id != i)
                    continue;

                Eigen::Vector2d uv =
                    use_noise ? obs.uv_noisy : obs.uv_gt;

                fout << uv.x() << " "
                     << uv.y() << " "
                     << obs.landmark_id
                     << endl;
            }

            fout.close();
        }
    }

    void saveMap(string folder, bool use_noise)
    {
        mkdir(folder.c_str(), 0777);

        for (auto &lm : landmarks)
        {
            stringstream ss;

            ss << folder << "/" << lm.id << ".txt";

            ofstream fout(ss.str());

            fout << lm.p_gt.x() << " "
                 << lm.p_gt.y() << " "
                 << lm.p_gt.z() << endl;

            for (auto &obs : observations)
            {
                if (obs.landmark_id != lm.id)
                    continue;

                Eigen::Vector2d uv =
                    use_noise ? obs.uv_noisy : obs.uv_gt;

                fout << obs.pose_id << " "
                     << obs.feature_id << " "
                     << uv.x() << " "
                     << uv.y() << endl;
            }

            fout.close();
        }
    }

    void saveTUM(string file, vector<Pose> &poses, double dt)
    {
        ofstream fout(file);

        for (int i = 0; i < poses.size(); i++)
        {
            double timestamp = i * dt;

            Eigen::Quaterniond q(poses[i].R);

            fout << fixed << setprecision(6)
                 << timestamp << " "
                 << poses[i].t.x() << " "
                 << poses[i].t.y() << " "
                 << poses[i].t.z() << " "
                 << q.x() << " "
                 << q.y() << " "
                 << q.z() << " "
                 << q.w() << endl;
        }

        fout.close();
    }

    void exportG2O(string file)
    {
        ofstream fout(file);

        for (int i = 0; i < poses_noisy.size(); i++)
        {
            Eigen::Quaterniond q(poses_noisy[i].R);

            fout << "VERTEX_SE3:QUAT " << i << " "
                 << poses_noisy[i].t.x() << " "
                 << poses_noisy[i].t.y() << " "
                 << poses_noisy[i].t.z() << " "
                 << q.x() << " "
                 << q.y() << " "
                 << q.z() << " "
                 << q.w() << endl;
        }

        int id_offset = poses_noisy.size();

        for (auto &lm : landmarks)
        {
            fout << "VERTEX_POINT_3D "
                 << id_offset + lm.id << " "
                 << lm.p_noisy.x() << " "
                 << lm.p_noisy.y() << " "
                 << lm.p_noisy.z() << endl;
        }

        fout.close();
    }

    void generate(int pose_num, int landmark_num)
    {
        generateTrajectory(pose_num);

        addPoseNoise();

        generateLandmarks(landmark_num);

        addLandmarkNoise();

        generateObservations();

        reset();
    }
};