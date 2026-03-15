#include <iostream>

#include "src/simulator.h"
#include "src/viewer.h"

using namespace std;

int main(int argc, char **argv)
{

    cout << "===== SLAM Simulator Start =====" << endl;

    // -----------------------------
    // 1 创建 Simulator
    // -----------------------------

    Simulator sim;

    int pose_num = 120;

    int landmark_num = 300;

    // 噪声参数（可自行调节）
    sim.pixel_noise_sigma = 1.0;

    sim.pose_noise_sigma = 0.05;

    sim.landmark_noise_sigma = 0.1;

    sim.outlier_ratio = 0.05;

    sim.dropout_ratio = 0.05;

    // -----------------------------
    // 2 生成模拟数据
    // -----------------------------

    sim.generate(pose_num, landmark_num);

    cout << "poses_gt: " << sim.poses_gt.size() << endl;
    cout << "poses_noisy: " << sim.poses_noisy.size() << endl;
    cout << "landmarks: " << sim.landmarks.size() << endl;
    cout << "observations: " << sim.observations.size() << endl;

    // -----------------------------
    // 3 保存轨迹 (TUM format)
    // -----------------------------

    double dt = 0.033;

    mkdir("data", 0777);
    mkdir("data/gt", 0777);
    mkdir("data/noisy", 0777);

    sim.saveFrames("data/gt/frames", dt, false);
    sim.saveMap("data/gt/maps", false);

    sim.saveFrames("data/noisy/frames", dt, true);
    sim.saveMap("data/noisy/maps", true);

    sim.saveTUM("data/trajectory_gt.txt",
                sim.poses_gt,
                0.1);

    sim.saveTUM("data/trajectory_noisy.txt",
                sim.poses_noisy,
                0.1);

    cout << "Trajectory saved (TUM format)" << endl;

    // -----------------------------
    // 4 导出 g2o 数据
    // -----------------------------

    sim.exportG2O("slam_graph.g2o");

    cout << "g2o graph exported" << endl;

    Frame frame;

    while (sim.getNextFrame(frame))
    {
        cout << "Frame "
             << frame.id
             << "  features "
             << frame.features.size()
             << endl;
        int count = 0;
        for (auto &f : frame.features)
        {
            cout << f.landmark_id << " "
                 << f.uv.transpose()
                 << endl;
        }
    }
    std::cout << "get frame end" << std::endl;
    // -----------------------------
    // 5 Viewer 可视化
    // -----------------------------

    Viewer viewer;

    viewer.setData(
        &sim.poses_gt,
        &sim.poses_noisy,
        &sim.landmarks,
        &sim.observations);

    viewer.run();

    return 0;
}