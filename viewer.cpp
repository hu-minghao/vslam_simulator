#include "viewer.h"
#include <iostream>

Viewer::Viewer()
{
    poses_ = nullptr;
    landmarks_ = nullptr;
    observations_ = nullptr;

    frame_id = 0;
    paused = false;
    show_rays = false;

    pangolin::CreateWindowAndBind("VSLAM Viewer", 1024, 768);

    glEnable(GL_DEPTH_TEST);

    s_cam = new pangolin::OpenGlRenderState(
        pangolin::ProjectionMatrix(1024, 768, 500, 500, 512, 389, 0.1, 1000),
        pangolin::ModelViewLookAt(0, -20, -20, 0, 0, 0, 0, -1, 0));

    d_cam = &pangolin::CreateDisplay()
                 .SetBounds(0, 1, 0, 1)
                 .SetHandler(new pangolin::Handler3D(*s_cam));

    pangolin::RegisterKeyPressCallback(' ', [this]()
                                       { paused = !paused; });

    pangolin::RegisterKeyPressCallback('n', [this]()
                                       { frame_id++; });

    pangolin::RegisterKeyPressCallback('b', [this]()
                                       {
        frame_id--;
        if(frame_id < 0) frame_id = 0; });

    pangolin::RegisterKeyPressCallback('r', [this]()
                                       { frame_id = 0; });

    pangolin::RegisterKeyPressCallback('l', [this]()
                                       { show_rays = !show_rays; });
}

void Viewer::setData(
    const std::vector<Pose> *poses,
    const std::vector<Landmark> *landmarks,
    const std::vector<Observation> *observations)
{
    poses_ = poses;
    landmarks_ = landmarks;
    observations_ = observations;
}

void Viewer::drawLandmarks()
{
    glPointSize(3);
    glColor3f(1, 0, 0);

    glBegin(GL_POINTS);

    for (auto &lm : *landmarks_)
    {
        glVertex3d(lm.p.x(), lm.p.y(), lm.p.z());
    }

    glEnd();
}

void Viewer::drawTrajectory()
{
    glLineWidth(2);
    glColor3f(0, 0, 1);

    glBegin(GL_LINE_STRIP);

    for (auto &p : *poses_)
    {
        glVertex3d(p.t.x(), p.t.y(), p.t.z());
    }

    glEnd();
}

void Viewer::drawCamera(const Pose &pose)
{
    const float w = 0.3;
    const float h = 0.2;
    const float z = 0.5;

    glPushMatrix();

    Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
    T.block<3, 3>(0, 0) = pose.R;
    T.block<3, 1>(0, 3) = pose.t;

    glMultMatrixd(T.data());

    glLineWidth(2);

    glColor3f(1, 1, 0);

    glBegin(GL_LINES);

    // optical center → image corners
    glVertex3f(0, 0, 0);
    glVertex3f(w, h, z);
    glVertex3f(0, 0, 0);
    glVertex3f(w, -h, z);
    glVertex3f(0, 0, 0);
    glVertex3f(-w, -h, z);
    glVertex3f(0, 0, 0);
    glVertex3f(-w, h, z);

    // image plane
    glVertex3f(w, h, z);
    glVertex3f(w, -h, z);
    glVertex3f(-w, h, z);
    glVertex3f(-w, -h, z);
    glVertex3f(-w, h, z);
    glVertex3f(w, h, z);
    glVertex3f(-w, -h, z);
    glVertex3f(w, -h, z);

    glEnd();

    // -------- camera axes --------
    float axis = 0.8;

    glLineWidth(3);
    glBegin(GL_LINES);

    // X axis
    glColor3f(1, 0, 0);
    glVertex3f(0, 0, 0);
    glVertex3f(axis, 0, 0);

    // Y axis
    glColor3f(0, 1, 0);
    glVertex3f(0, 0, 0);
    glVertex3f(0, axis, 0);

    // Z axis
    glColor3f(0, 0, 1);
    glVertex3f(0, 0, 0);
    glVertex3f(0, 0, axis);

    glEnd();

    glPopMatrix();
}

void Viewer::drawObservationRays()
{
    if (!show_rays)
        return;

    glLineWidth(1);
    glColor3f(0, 1, 1);

    const Pose &pose = (*poses_)[frame_id];

    glBegin(GL_LINES);

    for (auto &obs : *observations_)
    {
        if (obs.pose_id != frame_id)
            continue;

        const Landmark &lm = (*landmarks_)[obs.landmark_id];

        glVertex3d(pose.t.x(), pose.t.y(), pose.t.z());
        glVertex3d(lm.p.x(), lm.p.y(), lm.p.z());
    }

    glEnd();
}

void Viewer::drawImageFeatures()
{
    cv::Mat img(480, 640, CV_8UC3);
    img.setTo(0);

    for (auto &obs : *observations_)
    {
        if (obs.pose_id != frame_id)
            continue;

        cv::circle(img,
                   cv::Point(obs.uv.x(), obs.uv.y()),
                   3,
                   cv::Scalar(0, 255, 0),
                   -1);
    }

    cv::imshow("image", img);
}

void Viewer::run()
{
    while (!pangolin::ShouldQuit())
    {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        d_cam->Activate(*s_cam);

        drawLandmarks();

        drawTrajectory();

        drawCamera((*poses_)[frame_id]);

        drawObservationRays();

        pangolin::FinishFrame();

        drawImageFeatures();

        cv::waitKey(30);

        if (!paused)
        {
            frame_id++;

            if (frame_id >= poses_->size())
                frame_id = 0;
        }
    }
}