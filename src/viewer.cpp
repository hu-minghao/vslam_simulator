#include "viewer.h"
#include <iostream>

Viewer::Viewer()
{
    poses_gt_ = nullptr;
    poses_noisy_ = nullptr;
    landmarks_ = nullptr;
    observations_ = nullptr;

    frame_id = 0;

    paused = false;

    show_rays = false;

    show_gt_traj = true;

    show_noisy_traj = true;

    show_noisy_obs = false;

    show_noisy_landmark = false;

    show_error = false;

    pangolin::CreateWindowAndBind("VSLAM Viewer",1024,768);

    glEnable(GL_DEPTH_TEST);

    s_cam = new pangolin::OpenGlRenderState(
        pangolin::ProjectionMatrix(1024,768,500,500,512,389,0.1,1000),
        pangolin::ModelViewLookAt(0,-20,-20,0,0,0,0,-1,0));

    d_cam = &pangolin::CreateDisplay()
        .SetBounds(0,1,0,1)
        .SetHandler(new pangolin::Handler3D(*s_cam));


    pangolin::RegisterKeyPressCallback(' ',[this]()
    {
        paused = !paused;
    });

    pangolin::RegisterKeyPressCallback('n',[this]()
    {
        frame_id++;
    });

    pangolin::RegisterKeyPressCallback('b',[this]()
    {
        frame_id--;
        if(frame_id < 0) frame_id = 0;
    });

    pangolin::RegisterKeyPressCallback('r',[this]()
    {
        frame_id = 0;
    });

    pangolin::RegisterKeyPressCallback('l',[this]()
    {
        show_rays = !show_rays;
    });

    pangolin::RegisterKeyPressCallback('o',[this]()
    {
        show_noisy_obs = !show_noisy_obs;
    });

    pangolin::RegisterKeyPressCallback('p',[this]()
    {
        show_noisy_landmark = !show_noisy_landmark;
    });

    pangolin::RegisterKeyPressCallback('e',[this]()
    {
        show_error = !show_error;
    });

    pangolin::RegisterKeyPressCallback('g',[this]()
    {
        show_gt_traj = !show_gt_traj;
    });

    pangolin::RegisterKeyPressCallback('t',[this]()
    {
        show_noisy_traj = !show_noisy_traj;
    });
}

void Viewer::setData(
        const std::vector<Pose>* poses_gt,
        const std::vector<Pose>* poses_noisy,
        const std::vector<Landmark>* landmarks,
        const std::vector<Observation>* observations)
{
    poses_gt_ = poses_gt;

    poses_noisy_ = poses_noisy;

    landmarks_ = landmarks;

    observations_ = observations;
}

void Viewer::drawLandmarks()
{
    glPointSize(4);

    glBegin(GL_POINTS);

    for(const auto& lm : *landmarks_)
    {
        Eigen::Vector3d p;

        if(show_noisy_landmark)
        {
            p = lm.p_noisy;
            glColor3f(1,0.5,0);
        }
        else
        {
            p = lm.p_gt;
            glColor3f(1,0,0);
        }

        glVertex3d(p.x(),p.y(),p.z());
    }

    glEnd();
}

void Viewer::drawTrajectory()
{

    if(show_gt_traj && poses_gt_)
    {
        glLineWidth(3);

        glColor3f(0,1,0);

        glBegin(GL_LINE_STRIP);

        for(const auto& p : *poses_gt_)
        {
            glVertex3d(p.t.x(),p.t.y(),p.t.z());
        }

        glEnd();
    }

    if(show_noisy_traj && poses_noisy_)
    {
        glLineWidth(2);

        glColor3f(1,0,0);

        glBegin(GL_LINE_STRIP);

        for(const auto& p : *poses_noisy_)
        {
            glVertex3d(p.t.x(),p.t.y(),p.t.z());
        }

        glEnd();
    }
}

void Viewer::drawCamera(const Pose& pose)
{
    const float w = 0.3;
    const float h = 0.2;
    const float z = 0.5;

    glPushMatrix();

    Eigen::Matrix4d T = Eigen::Matrix4d::Identity();

    T.block<3,3>(0,0) = pose.R;

    T.block<3,1>(0,3) = pose.t;

    glMultMatrixd(T.data());

    glLineWidth(2);

    glColor3f(1,1,0);

    glBegin(GL_LINES);

    glVertex3f(0,0,0); glVertex3f(w,h,z);
    glVertex3f(0,0,0); glVertex3f(w,-h,z);
    glVertex3f(0,0,0); glVertex3f(-w,-h,z);
    glVertex3f(0,0,0); glVertex3f(-w,h,z);

    glVertex3f(w,h,z); glVertex3f(w,-h,z);
    glVertex3f(-w,h,z); glVertex3f(-w,-h,z);
    glVertex3f(-w,h,z); glVertex3f(w,h,z);
    glVertex3f(-w,-h,z); glVertex3f(w,-h,z);

    glEnd();

    float axis = 0.8;

    glLineWidth(3);

    glBegin(GL_LINES);

    glColor3f(1,0,0);
    glVertex3f(0,0,0);
    glVertex3f(axis,0,0);

    glColor3f(0,1,0);
    glVertex3f(0,0,0);
    glVertex3f(0,axis,0);

    glColor3f(0,0,1);
    glVertex3f(0,0,0);
    glVertex3f(0,0,axis);

    glEnd();

    glPopMatrix();
}

void Viewer::drawObservationRays()
{
    if(!show_rays) return;

    const Pose& pose = (*poses_noisy_)[frame_id];

    glLineWidth(1);

    glColor3f(0,1,1);

    glBegin(GL_LINES);

    for(const auto& obs : *observations_)
    {
        if(obs.pose_id != frame_id) continue;

        const Landmark& lm = (*landmarks_)[obs.landmark_id];

        Eigen::Vector3d p;

        if(show_noisy_landmark)
            p = lm.p_noisy;
        else
            p = lm.p_gt;

        glVertex3d(pose.t.x(),pose.t.y(),pose.t.z());

        glVertex3d(p.x(),p.y(),p.z());
    }

    glEnd();
}

void Viewer::drawImageFeatures()
{
    cv::Mat img(480,640,CV_8UC3);

    img.setTo(0);

    for(const auto& obs : *observations_)
    {
        if(obs.pose_id != frame_id) continue;

        Eigen::Vector2d uv;

        if(show_noisy_obs)
            uv = obs.uv_noisy;
        else
            uv = obs.uv_gt;

        cv::Scalar color(0,255,0);

        if(obs.is_outlier)
            color = cv::Scalar(0,0,255);

        cv::circle(img,
                   cv::Point(uv.x(),uv.y()),
                   3,
                   color,
                   -1);

        if(show_error)
        {
            cv::line(img,
                     cv::Point(obs.uv_gt.x(),obs.uv_gt.y()),
                     cv::Point(obs.uv_noisy.x(),obs.uv_noisy.y()),
                     cv::Scalar(255,0,0));
        }
    }

    cv::imshow("image",img);
}

void Viewer::run()
{
    while(!pangolin::ShouldQuit())
    {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        d_cam->Activate(*s_cam);

        drawLandmarks();

        drawTrajectory();

        drawCamera((*poses_gt_)[frame_id]);

        drawObservationRays();

        pangolin::FinishFrame();

        drawImageFeatures();

        cv::waitKey(30);

        if(!paused)
        {
            frame_id++;

            if(frame_id >= poses_gt_->size())
                frame_id = 0;
        }
    }
}