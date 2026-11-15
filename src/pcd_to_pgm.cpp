#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <nav_msgs/msg/occupancy_grid.hpp>
#include <ament_index_cpp/get_package_share_directory.hpp>//ament_index_cpp::get_package_share_directory("pcd_to_pgm_demo")查找配置的点云文件

#include <pcl/io/pcd_io.h>//pcl::io::loadPCDFile<pcl::PointXYZ>()读取点云文件
#include <pcl/point_cloud.h>//定义模板类pcl::PointCloud<T>
#include <pcl/point_types.h>//点云类型pcl::PointXYZ
#include <pcl/filters/passthrough.h>//pcl::PassThrough<pcl::PointXYZ>直通滤波
#include <pcl/filters/filter.h>//pcl::removeNaNFromPointCloud()
#include <pcl/common/io.h>//pcl::copyPointCloud
#include <pcl_conversions/pcl_conversions.h>//pcl::toROSMsg(cloud, msg)转换成ros2点云消息

#include <algorithm>
#include <cmath>//std::nextafter生成开区间
#include <cstdint>//定长整形uint_32
#include <filesystem>
//fs::exists检查文件路径，fs::create_directories创建输出目录，fs::path拼接路径
#include <fstream>//读写文件流
#include <iomanip>//std::fixed 和 std::setprecision(9)，控制配置参数的浮点数
#include <limits>//std::numeric_limits<float>::infinity()作为std::nextafter的参数配合构造开区间
#include <stdexcept>//std::runtime_error
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace
{

using PointCloud = pcl::PointCloud<pcl::PointXYZ>;

//读取点云文件
PointCloud::Ptr loadCloud(const std::string & path)
{
  //创建点云
  auto cloud = std::make_shared<PointCloud>();
  //
  if (pcl::io::loadPCDFile<pcl::PointXYZ>(path, *cloud) < 0)//返回0 表示成功，小于0表示失败 
  {
    throw std::runtime_error("读取点云失败: " + path);
  }
  if (cloud->empty()) {
    throw std::runtime_error("点云为空: " + path);
  }

  // 丢掉 x/y/z 任一为 NaN / Inf 的点，否则后面 floor() 会得到未定义结果。
  cloud->is_dense = false;
  std::vector<int> valid;
  pcl::removeNaNFromPointCloud(*cloud, *cloud, valid);
  if (cloud->empty()) 
  {
    throw std::runtime_error("点云中没有有效点: " + path);
  }
  return cloud;
}

// 高度筛选
PointCloud::Ptr filterByHeight(const PointCloud::Ptr & cloud, float zlo, float zhi)
{
  //创建新点云
  auto filtered = std::make_shared<PointCloud>();
  //直通滤波
  pcl::PassThrough<pcl::PointXYZ> pass;
  //放入待处理对象
  pass.setInputCloud(cloud);
  //以z轴为过滤
  pass.setFilterFieldName("z");
  //设置过滤边界
  pass.setFilterLimits(
    //向正无穷的方向寻找zlo的下一个浮点数，形成浮点开区间
    std::nextafter(zlo, std::numeric_limits<float>::infinity()),
    std::nextafter(zhi, -std::numeric_limits<float>::infinity()));//得到zlo < z < zhi
  //执行滤波，将合理点存入点云
  pass.filter(*filtered);
  return filtered;
}

sensor_msgs::msg::PointCloud2 toCloud2(
  const PointCloud & cloud, const std::string & frame, const rclcpp::Time & stamp)
{
  sensor_msgs::msg::PointCloud2 msg;
  //将pcl点云转换为ros2点云消息
  pcl::toROSMsg(cloud, msg);
  //设置header
  msg.header.frame_id = frame;
  msg.header.stamp = stamp;
  return msg;
}

nav_msgs::msg::OccupancyGrid toGrid(
  const PointCloud & cloud, double resolution, const std::string & frame)
{
  if (cloud.empty()) 
  {
    throw std::runtime_error("筛选后没有点，检查 z_min / z_max");
  }

  //用第一个点初始化极值
  float x0 = cloud.front().x;
  float y0 = cloud.front().y;
  //用第一个点初始化第二个点
  float x1 = x0;
  float y1 = y0;
  //用两个点遍历点云找出坐标最小点x0.y0和坐标最大点x1.y1，得到包裹点云的最小矩形
  for (const auto & p : cloud) 
  {
    x0 = std::min(x0, p.x);
    y0 = std::min(y0, p.y);
    x1 = std::max(x1, p.x);
    y1 = std::max(y1, p.y);
  }

  // 原点向下对齐到 resolution 的整数倍，保证 (x - ox) / res >= 0
  const double ox = std::floor(static_cast<double>(x0) / resolution) * resolution;
  const double oy = std::floor(static_cast<double>(y0) / resolution) * resolution;
  //矩形的宽高，要用坐标差除分辨率加一，因为下标索引从0开始，宽高从1开始
  const auto width = static_cast<uint32_t>(std::floor((x1 - ox) / resolution)) + 1;
  const auto height = static_cast<uint32_t>(std::floor((y1 - oy) / resolution)) + 1;

  //初始化栅格消息
  nav_msgs::msg::OccupancyGrid grid;
  grid.header.frame_id = frame;
  grid.info.resolution = static_cast<float>(resolution);
  grid.info.width = width;
  grid.info.height = height;
  grid.info.origin.position.x = ox;
  grid.info.origin.position.y = oy;
  grid.info.origin.position.z = 0.0;
  grid.info.origin.orientation.w = 1.0;
  //初始化空闲栅格
  grid.data.assign(static_cast<size_t>(width) * height, 0);

  //遍历点云转换为栅格
  for (const auto & p : cloud) 
  {
    //点云的二维栅格坐标
    const auto gx = static_cast<uint32_t>(std::floor((p.x - ox) / resolution));
    const auto gy = static_cast<uint32_t>(std::floor((p.y - oy) / resolution));
    if (gx >= width || gy >= height) 
    {
      continue;
    }
    //占据
    grid.data[static_cast<size_t>(gy) * width + gx] = 100;
  }
  return grid;
}

//将栅格地图保存
void saveMap(const nav_msgs::msg::OccupancyGrid & grid, const std::string & output_dir)
{
  //创建输出目录
  fs::create_directories(output_dir);

  //fs::path(output_dir) / "map.pgm" 将路径拼接起来
  const std::string pgm_path = (fs::path(output_dir) / "map.pgm").string();//在此保存地图的二维灰度图像
  std::ofstream pgm(pgm_path, std::ios::binary);//用二进制的方式打开并返回是否打开成功
  if (!pgm) {
    throw std::runtime_error("无法写入 " + pgm_path);
  }
  //P5 表示二进制灰度图，接着写入宽、空格隔开，高，255 是最大灰度值
  pgm << "P5\n" << grid.info.width << " " << grid.info.height << "\n255\n";
  // PGM 第 0 行是图像最上方（世界 y 最大），所以栅格行要倒着写
  for (int y = static_cast<int>(grid.info.height) - 1; y >= 0; --y)//栅格索引从零开始，是高减一
  {
    for (uint32_t x = 0; x < grid.info.width; ++x) 
    {
      //size_t是无符号整数类型，最小值为0
      const int v = grid.data[static_cast<size_t>(y) * grid.info.width + x];
      //遍历栅格地图并输出
      pgm.put(static_cast<char>(v ? 0 : 254));
    }
  }
  pgm.close();

  //nav2地图必须配置参数：图像文件名、分辨率、地图左下角在世界坐标系中的真实位置 origin，以及占据/空闲的阈值判定标准
  const std::string yaml_path = (fs::path(output_dir) / "map.yaml").string();//在此保存地图的元数据
  std::ofstream yaml(yaml_path);
  if (!yaml) {
    throw std::runtime_error("无法写入 " + yaml_path);
  }
  //fixed强制流以定点表示法表示浮点数，保留9位小数
  yaml << std::fixed << std::setprecision(9);
  yaml << "image: map.pgm\n"//图像文件路径
       << "resolution: " << grid.info.resolution << "\n"
       << "origin: [" << grid.info.origin.position.x << ", "
       << grid.info.origin.position.y << ", 0.0]\n"
       << "negate: 0\n"
       << "occupied_thresh: 0.65\n"
       << "free_thresh: 0.196\n";
  yaml.close();
}

std::string resolvePcdPath(const std::string & configured)
{
  //直接找到路径
  if (fs::exists(configured)) 
  {
    return configured;
  }

  try 
  {
    //get_package_share_directory 会去 install/share/目录下寻找指定的包，并返回该包安装后的绝对路径
    const fs::path share = ament_index_cpp::get_package_share_directory("pcd_to_pgm_demo");
    const fs::path direct = share / configured;
    //绝对路径
    if (fs::exists(direct)) 
    {
      return direct.string();
    }
    //剥离掉所有路径前缀，只保留文件名，拼接到 share/data/ 目录下查找
    const fs::path in_data = share / "data" / fs::path(configured).filename();
    if (fs::exists(in_data)) 
    {
      return in_data.string();
    }
  } 
  //静默处理
  catch (const std::exception &) 
  {
    // 包还没安装时 get_package_share_directory 会抛异常，退回配置值
  }
  return configured;
}

}  // namespace

class PcdToPgm : public rclcpp::Node
{
public:
  PcdToPgm() : Node("pcd_to_pgm")
  {
    //参数
    const auto pcd_file = declare_parameter<std::string>("pcd_file", "data/rmul_2025.pcd");
    const auto output_dir = declare_parameter<std::string>("output_dir", "output");
    const auto res = declare_parameter<double>("resolution", 0.05);
    const auto frame_id = declare_parameter<std::string>("frame_id", "map");
    const auto save_map = declare_parameter<bool>("save_map", true);
    const float zlo = static_cast<float>(declare_parameter<double>("z_min", 0.1));
    const float zhi = static_cast<float>(declare_parameter<double>("z_max", 1.5));

    //分辨率合理性
    if (res <= 0.0) 
    {
      throw std::runtime_error("resolution 必须大于 0");
    }

	//得到文件路径
    const std::string path = resolvePcdPath(pcd_file);
    //读取点云文件
    const auto cloud = loadCloud(path);

    //用最前面的点初始化两个点云高度
    float zmin_all = cloud->front().z;
    float zmax_all = cloud->front().z;
    //遍历点云得到高度最大和最小值
    for (const auto & p : *cloud) {
      zmin_all = std::min(zmin_all, p.z);
      zmax_all = std::max(zmax_all, p.z);
    }
    //反馈
    RCLCPP_INFO(
      get_logger(), "读取点云: %s (%zu 点, z %.3f ~ %.3f)",
      path.c_str(), cloud->size(), zmin_all, zmax_all);

    //高度筛选后的点云
    const auto filtered = filterByHeight(cloud, zlo, zhi);
    //反馈
    RCLCPP_INFO(
      get_logger(), "高度筛选 %.2f < z < %.2f: 保留 %zu / %zu 点",
      zlo, zhi, filtered->size(), cloud->size());

	//高度筛选之后的点云转换为二维栅格地图
    auto grid = toGrid(*filtered, res, frame_id);
    //设置时间戳
    grid.header.stamp = now();

	//是否保存地图
    if (save_map) 
    {
      saveMap(grid, output_dir);
      RCLCPP_INFO(
        get_logger(), "已导出 %s/map.pgm 和 %s/map.yaml",
        output_dir.c_str(), output_dir.c_str());
    }

	//设置消息频率
    const auto qos = rclcpp::QoS(1).transient_local();
    //创建发布方
    pub_cloud_ = create_publisher<sensor_msgs::msg::PointCloud2>("/cloud", qos);
    pub_filtered_ = create_publisher<sensor_msgs::msg::PointCloud2>("/cloud_filtered", qos);
    pub_map_ = create_publisher<nav_msgs::msg::OccupancyGrid>("/map", qos);

	//发布消息
    pub_cloud_->publish(toCloud2(*cloud, frame_id, grid.header.stamp));
    pub_filtered_->publish(toCloud2(*filtered, frame_id, grid.header.stamp));
    pub_map_->publish(grid);

	//反馈
    RCLCPP_INFO(
      get_logger(), "地图 %ux%u, 分辨率 %.3f m, 原点 (%.3f, %.3f)",
      grid.info.width, grid.info.height, grid.info.resolution,
      grid.info.origin.position.x, grid.info.origin.position.y);
  }

private:
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_cloud_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_filtered_;
  rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr pub_map_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  try {
    rclcpp::spin(std::make_shared<PcdToPgm>());
  } catch (const std::exception & e) {
    RCLCPP_ERROR(rclcpp::get_logger("pcd_to_pgm"), "发生异常: %s", e.what());
    rclcpp::shutdown();
    return 1;
  }
  rclcpp::shutdown();
  return 0;
}
