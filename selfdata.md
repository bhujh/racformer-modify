目的：使用racformer的网络模型加载已有模型权重文件checkpoints/racformer，以测试模式测试我们自己采集的radar和images数据，进行3D目标检测。


传感器配置：一个radar传感器，一个相机传感器
已采集的数据：radar,images,相机内参，radar到camera的外参


要求：
    1.创建own_datdaset.py文件，编写自定义数据集类代码加载自己采集的数据；
    2.使得传送给模型入口的数据组织形式和nuscenes数据相同，包括图像数据、雷达点云、雷达深度图、雷达RCS图、元数据。但是元数据没有sample token，历史帧sweeps_num的数量也不同;
    3.使用相同的测试模式数据增强方式;
    4.创建configs/own_racformer_r50_nuimg_704x256_f8.py，模型和数据配置参考configs/racformer_r50_nuimg_704x256_f8.py


备注：
    1.要顺利加载数据，传入推理模型中，是否必须使用mmdet的功能。
    2.还有什么要求，可以再完善。