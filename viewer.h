#pragma once

#include <pangolin/pangolin.h>
#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <vector>
#include "type_def.h"

class Viewer
{
public:

    Viewer();

    void setData(
        const std::vector<Pose>* poses,
        const std::vector<Landmark>* landmarks,
        const std::vector<Observation>* observations);

    void run();

private:

    void drawLandmarks();
    void drawTrajectory();
    void drawCamera(const Pose &pose);
    void drawObservationRays();
    void drawImageFeatures();

private:

    const std::vector<Pose>* poses_;
    const std::vector<Landmark>* landmarks_;
    const std::vector<Observation>* observations_;

    int frame_id;

    bool paused;
    bool show_rays;

    pangolin::OpenGlRenderState* s_cam;
    pangolin::View* d_cam;
};