#pragma once

#include <Eigen/Dense>
#include <vector>
#include <pangolin/pangolin.h>
#include <opencv2/opencv.hpp>

#include "type_def.h"

class Viewer
{
public:

    Viewer();

    void setData(
        const std::vector<Pose>* poses_gt,
        const std::vector<Pose>* poses_noisy,
        const std::vector<Landmark>* landmarks,
        const std::vector<Observation>* observations);

    void run();

private:

    const std::vector<Pose>* poses_gt_;
    const std::vector<Pose>* poses_noisy_;

    const std::vector<Landmark>* landmarks_;
    const std::vector<Observation>* observations_;

    pangolin::OpenGlRenderState* s_cam;
    pangolin::View* d_cam;

    int frame_id;

    bool paused;

    bool show_rays;

    bool show_gt_traj;

    bool show_noisy_traj;

    bool show_noisy_obs;

    bool show_noisy_landmark;

    bool show_error;

private:

    void drawLandmarks();

    void drawTrajectory();

    void drawCamera(const Pose& pose);

    void drawObservationRays();

    void drawImageFeatures();
};