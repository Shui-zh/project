#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>
#include <pcl_ros/transforms.hpp> // 用于 TF 转换点云

class RealTimeMapper : public rclcpp::Node {
public:
    RealTimeMapper() : Node("real_time_mapper") {
        // 1. 初始化 TF 监听器 (极其重要，否则雷达数据无法对齐)
        tf_buffer_ = std::make_shared<tf2_ros::Buffer>(this->get_clock());
        tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

        // 2. 订阅 3D 雷达点云
        sub_cloud_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            "/points_raw", rclcpp::SensorDataQoS(),
            std::bind(&RealTimeMapper::cloudCallback, this, std::placeholders::_1));

        // 3. 发布 2D 栅格地图
        pub_map_ = this->create_publisher<nav_msgs::msg::OccupancyGrid>("/map", 10);

        // 4. 初始化全局地图 (例如：以 0,0 为中心，50m x 50m 的地图)
        initGlobalMap(50.0, 50.0, 0.05); // 宽50m, 高50m, 分辨率5cm
    }

private:
    void cloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
        // A. 将 ROS 消息转为 PCL 格式
        PointCloud::Ptr cloud(new PointCloud);
        pcl::fromROSMsg(*msg, *cloud);

        // B. TF 坐标变换：将点云从雷达坐标系转换到 map 坐标系
        PointCloud::Ptr cloud_map(new PointCloud);
        try {
            // 等待并获取从 map 到 雷达frame 的变换
            auto transform = tf_buffer_->lookupTransform(
                "map", msg->header.frame_id, msg->header.stamp, rclcpp::Duration::from_seconds(0.1));
            pcl_ros::transformPointCloud(*cloud, *cloud_map, transform);
        } catch (tf2::TransformException &ex) {
            RCLCPP_WARN(this->get_logger(), "TF 获取失败: %s", ex.what());
            return; // 变换失败就丢弃这一帧
        }

        // C. 高度过滤 (直接复用你之前的逻辑，但稍微改一下)
        PointCloud::Ptr filtered(new PointCloud);
        pcl::PassThrough<pcl::PointXYZ> pass;
        pass.setInputCloud(cloud_map);
        pass.setFilterFieldName("z");
        pass.setFilterLimits(0.1, 1.5); // 过滤掉地面和过高的障碍
        pass.filter(*filtered);

        // D. 将点云映射到全局栅格数组中
        updateGlobalGrid(*filtered);

        // E. 发布地图
        publishMap();
    }

    void initGlobalMap(double width_m, double height_m, double res) {
        resolution_ = res;
        width_ = static_cast<uint32_t>(width_m / res);
        height_ = static_cast<uint32_t>(height_m / res);
        // 原点设在物理坐标的负偏移处，使得 (0,0) 在地图中心
        origin_x_ = -width_m / 2.0; 
        origin_y_ = -height_m / 2.0;
        
        // 初始化为未知 (-1) 或空闲 (0)
        global_grid_.assign(width_ * height_, 0); 
    }

    void updateGlobalGrid(const PointCloud &cloud) {
        for (const auto &p : cloud) {
            // 计算在全局栅格中的索引
            int gx = std::floor((p.x - origin_x_) / resolution_);
            int gy = std::floor((p.y - origin_y_) / resolution_);
            
            // 边界检查
            if (gx >= 0 && gx < width_ && gy >= 0 && gy < height_) {
                // 标记为占据 (100)
                global_grid_[gy * width_ + gx] = 100; 
            }
        }
    }

    void publishMap() {
        nav_msgs::msg::OccupancyGrid grid;
        grid.header.frame_id = "map";
        grid.header.stamp = this->now();
        grid.info.resolution = resolution_;
        grid.info.width = width_;
        grid.info.height = height_;
        grid.info.origin.position.x = origin_x_;
        grid.info.origin.position.y = origin_y_;
        grid.info.origin.orientation.w = 1.0;
        grid.data = global_grid_;
        pub_map_->publish(grid);
    }

    // 变量声明
    std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_cloud_;
    rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr pub_map_;
    
    double resolution_;
    uint32_t width_, height_;
    double origin_x_, origin_y_;
    std::vector<int8_t> global_grid_;
};
