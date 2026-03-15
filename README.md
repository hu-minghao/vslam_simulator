## observation
绿色为内点观测
红色为外点观测

## data
模拟数据需要生成真值，包括轨迹真值及三维点的真值
观测则是分：不带噪声观测，加入噪声观测
然后基于观测，看恢复出来的三维点及轨迹是否合理
但为了验证优化问题，三维点及轨迹带噪声也方便看恢复的点是否正常

frame:
0
8 0 3 0 0 0 1
244.104 386.144 17

frame_id
x y z qx qy qz qw
u v landmark_id

map:
3.33713 -3.70602 10.045
90 0 623.049 2.94483
x y z
frame_id feature_id u v


## Commond
space   pause
n       next frame
b       previous
r       reset

g       toggle GT trajectory
t       toggle noisy trajectory

o       toggle noisy feature
p       toggle noisy landmark

l       show observation rays
e       reprojection error