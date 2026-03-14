#include "simulator.h"
#include "viewer.h"
#include <iostream>

int main()
{

    Simulator sim;

    sim.generate(
        80,   // pose数量
        200   // landmark数量
    );

    std::cout<<"poses: "<<sim.poses.size()<<std::endl;
    std::cout<<"landmarks: "<<sim.landmarks.size()<<std::endl;
    std::cout<<"observations: "<<sim.observations.size()<<std::endl;

    Viewer viewer;

    viewer.setData(
        &sim.poses,
        &sim.landmarks,
        &sim.observations
    );

    viewer.run();

    return 0;
}