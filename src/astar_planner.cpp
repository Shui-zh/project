#include "rclcpp/rclcpp.hpp"
#include "geometry_msgs/msg/pose_with_covariance_stamped.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "nav_msgs/msg/path.hpp"
#include "nav_msgs/msg/occupancy_grid.hpp"

#include <chrono>
#include <cstdint>
#include <cmath>
#include <queue>
#include <cstring>
#include <vector>
#include <algorithm>

//栅格
struct Cell
{
    int x;
    int y;
};

//栅格地图
struct Grid
{
    std::vector<int8_t> data;//是否占据0,100
    int width;
    int height;
    double res;
    double ox = 0.0;
    double oy = 0.0;
};

//栅格路径
struct Result
{
    bool found = false;
    int expended = 0;	//扩展节点数
    std::vector<Cell> path;	//保存栅格形成的路径 
};

//八领域计算代价
double heuristic(int dx, int dy)
{
    // 八邻域 octile：max + (√2-1)*min
    const int adx = std::abs(dx);
    const int ady = std::abs(dy);
    const int mn = std::min(adx, ady);
    const int mx = std::max(adx, ady);
    return mx + (std::sqrt(2.0) - 1.0) * mn;
}

//A*
Result planAstar(const Grid &g, const Cell &start, const Cell &goal, int thresh, bool allow_diagonal)
{
    Result r;
    const int w = g.width;
    const int h = g.height;
    //n是栅格总数
    const int n = w * h;

    //某格能不能走：不越界且灰度低于阈值
    const auto freeCell = [&] ( int cx, int cy)//引用方式传入参数
    {
        if(cx >= w || cy >= h || cx < 0 || cy < 0)
        {
            return false;
        }
        return g.data[cy * w + cx] < thresh;
    };

    //起点或终点是否越界
    if(!freeCell(start.x, start.y) || !freeCell(goal.x, goal.y))
    {
        return r;
    }

    //inf等于无穷大
    const double INF = 1e18;
    //起点到每个格子的最短代价，先默认填入无穷大
    std::vector<double> gscore(n, INF);
    //每个格子的父格子，用于回溯路径
    std::vector<int> parent(n, -1);
    //标记已经扩展的格子
    std::vector<char> closed(n, 0);

    //起点和终点转换为一维栅格坐标
    const int si = start.y * w + start.x;
    const int gi = goal.y * w + goal.x;
    gscore[si] = 0.0;

	//double代价和int一维栅格坐标
    using Item = std::pair<double, int>;
    //小顶堆
    std::priority_queue<Item, std::vector<Item>, std::greater<Item>> open;
    open.push({heuristic(goal.x - start.x, goal.y - start.y), si});

    //八向
    const int dx[8] = {1, -1, 0, 0, 1, 1, -1, -1};
    const int dy[8] = {0, 0, 1, -1, 1, -1, 1, -1};

    while(!open.empty())
    {
        const int cur = open.top().second;//代价最小的栅格坐标
        open.pop();
        //排除已经被扩展
        if(closed[cur])
        {
            continue;
        }
        //标记为已扩展
        closed[cur] = 1;
        r.expended++;
        if(cur == gi)
        {
            break;//排除终点
        }
		
		//一维栅格坐标转换
        const int cx = cur % w;
        const int cy = cur / w;

		//计算往每个方向走的代价
        for(int k = 0;k < 8;k++)//k代表8个方向
        {
            if(k >= 4 && !allow_diagonal)
            {
                continue;//
            }
            //走了一步之后的坐标
            const int nx = cx + dx[k];
            const int ny = cy + dy[k];
            if(!freeCell(nx, ny))
            {
                continue;
            }
            //是不是走斜线
            const bool diagonal = (k >= 4);
            //斜角的栅格是否为空
            if(diagonal && (!freeCell(cx + dx[k], cy) || !freeCell(cx, cy + dy[k])))
            {
                continue;
            }
            //将移动之后的新坐标转换成一维
            const int ni = ny * w + nx;
            //目标栅格是否已扩展
            if(closed[ni])
            {
                continue;
            }
            //这一步的移动代价
            const double step = diagonal ? std::sqrt(2.0) : 1.0;
            //旧代价加上移动代价
            const double ng = gscore[cur] + step;
            //如果代价比原来的小就替换
            if(ng < gscore[ni])
            {
                gscore[ni] = ng;
                //记录父栅格坐标
                parent[ni] = cur;
                //A*算法（总代价加上此栅格到终点的代价
                open.push({ng + heuristic(goal.x - nx, goal.y - ny), ni});
            }
        }
    }

	//终点的代价还是默认无穷大，说明没找到路
    if(gscore[gi] >= INF)
    {
        return r;
    }

    //从终点开始回溯途径栅格点并放入路径
    for(int i = gi;i != -1;i = parent[i])
    {
        r.path.push_back(Cell{i % w, i / w});
    }
    
    //将路径翻转变回起点到终点
    std::reverse(r.path.begin(), r.path.end());
    r.found = true;
    return r;
}

class AstarPlanner : public rclcpp::Node
{
public:
    AstarPlanner(): Node("astar")
    {
        start_x_ = declare_parameter<double>("start_x", 1.0);
        start_y_ = declare_parameter<double>("start_y", 1.0);
        goal_x_ = declare_parameter<double>("goal_x", 7.0);
        goal_y_ = declare_parameter<double>("goal_y", 11.0); 
        thresh_ = declare_parameter<int>("occupied_thresh", 50);
        frame_id_ = declare_parameter<std::string>("frame_id", "map");
        allow_diagonal_ = declare_parameter<bool>("allow_diagonal", true);
        inflation_cells_ = declare_parameter<int>("inflation_cells", 4);

        //路径发布
        pub_path_ = create_publisher<nav_msgs::msg::Path>("/path", rclcpp::QoS(1).transient_local());
    
        //栅格地图接收
        sub_map_ = create_subscription<nav_msgs::msg::OccupancyGrid>("/map", rclcpp::QoS(1).transient_local(), 
            [this] (const nav_msgs::msg::OccupancyGrid::SharedPtr msg) { onMap(*msg); });

        //起点接收
        sub_start_ = create_subscription<geometry_msgs::msg::PoseWithCovarianceStamped>("/initialpose", rclcpp::QoS(10),
            [this] (const geometry_msgs::msg::PoseWithCovarianceStamped::SharedPtr msg) 
            {
                start_x_ = msg->pose.pose.position.x;
                start_y_ = msg->pose.pose.position.y;
                plan();
            });

        //终点接收
        sub_goal_ = create_subscription<geometry_msgs::msg::PoseStamped>("/goal_pose", rclcpp::QoS(10),
            [this] (const geometry_msgs::msg::PoseStamped::SharedPtr msg) 
            {
                goal_x_ = msg->pose.position.x;
                goal_y_ = msg->pose.position.y;
                plan();
            });

        //回馈
        RCLCPP_INFO(get_logger(), "A* 就绪：起点 (%.2f, %.2f) 终点 (%.2f, %.2f)，等 /map", 
            start_x_, start_y_, goal_x_, goal_y_);
    }
private:
    //将栅格地图消息转换出地图
    void onMap(const nav_msgs::msg::OccupancyGrid &msg)
    {   
        grid_.width = static_cast<int>(msg.info.width);
        grid_.height = static_cast<int>(msg.info.height);
        grid_.ox = msg.info.origin.position.x;
        grid_.oy = msg.info.origin.position.y;
        grid_.res = msg.info.resolution;
        grid_.data = msg.data;
       
        if(!msg.header.frame_id.empty())
        {
            frame_id_ = msg.header.frame_id;
        }
        has_map_ = true;
        RCLCPP_INFO(this->get_logger(), "Receive map: %dx%d @ %.2fm, origin(%.2f, %.2f)", 
            grid_.width, grid_.height, grid_.res, grid_.ox, grid_.oy);
        //
        plan();
    }

    //转换为栅格
    Cell worldToCell(double x, double y) const
    {
        return Cell{
            static_cast<int>(std::floor((x - grid_.ox) / grid_.res)),
            static_cast<int>(std::floor((y - grid_.oy) / grid_.res))};
    }

    //
    bool isFree(const Cell &c) const
    {
    	//越界
        if(c.x < 0 || c.y < 0 || c.x >= grid_.width || c.y >= grid_.height)
        {
            return false;
        }
        //空闲
        return grid_.data[c.y * grid_.width + c.x] < thresh_;
    }
 
    //规划路径
    void plan()
    {
        if(!has_map_)
        {
            RCLCPP_WARN(this->get_logger(), "还没收到 /map,等任务一发布地图");
            return;
        }
		
		//终点和起点转换为栅格
        const Cell s = worldToCell(start_x_, start_y_);
        const Cell g = worldToCell(goal_x_, goal_y_);
        if(!isFree(s))
        {
            clearPath("起点越界或落在障碍上");
            return;
        }
        if(!isFree(g))
        {
            clearPath("终点越界或落在障碍上");
            return;
        }
        //起始时间戳
        const auto t0 = std::chrono::steady_clock::now();
        const Result r = planAstar(grid_, s, g, thresh_, allow_diagonal_);
        //处理消耗的时间
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0 ).count();

        if(!r.found)
        {
            clearPath("没有可达路径");
            return;
        }
    
        //转换为路径消息
        nav_msgs::msg::Path path;
        path.header.frame_id = frame_id_;
        path.header.stamp = now();
        path.poses.reserve(r.path.size());
        for(const auto &c : r.path)
        {
            geometry_msgs::msg::PoseStamped ps;
            ps.header = path.header;
            //栅格路径点转换为米的点消息
            ps.pose.position.x = grid_.ox + (c.x + 0.5) * grid_.res;
            ps.pose.position.y = grid_.oy + (c.y + 0.5) * grid_.res;
            ps.pose.position.z = 0.0;
            ps.pose.orientation.w = 1.0;
            path.poses.push_back(ps);
        }
        pub_path_->publish(path);

        //路径总长
        double len = 0.0;
        for(std::size_t i = 1;i < r.path.size();i++)
        {
            const double ddx = r.path[i].x - r.path[i - 1].x;
            const double ddy = r.path[i].y - r.path[i - 1].y;
            len += std::sqrt(ddx * ddx + ddy * ddy) * grid_.res;
        }
        RCLCPP_INFO(get_logger(), "路径 %zu 格, 长度 %.2fm, 扩展 %d 格, 耗时 %.1fms",
            r.path.size(), len, r.expended, ms);
    }
    
    //清除上一条路径
    void clearPath(const std::string &why)
    {
        nav_msgs::msg::Path empty;
        empty.header.frame_id = frame_id_;
        empty.header.stamp = now();
        //发布空路径覆盖
        pub_path_->publish(empty);
        RCLCPP_WARN(this->get_logger(), "%s,已清除旧路径", why.c_str());
    }

    Grid grid_;
    bool has_map_ = false;
    int thresh_ = 50;
    double start_x_ = 1.0, start_y_ = 1.0;
    double goal_x_ = 7.0, goal_y_ = 11.0;
    std::string frame_id_ = "map";
    bool allow_diagonal_ = true;
    int inflation_cells_ = 4;

    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr pub_path_;
    rclcpp::Subscription<nav_msgs::msg::OccupancyGrid>::SharedPtr sub_map_;
    rclcpp::Subscription<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr sub_start_;
    rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr sub_goal_;
};

int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);
    try
    {
        rclcpp::spin(std::make_shared<AstarPlanner>());
    }
    catch(const std::exception& ep)
    {
        RCLCPP_ERROR(rclcpp::get_logger("astar"), "error: %s", ep.what());
        rclcpp::shutdown();
        return 1;
    }
    rclcpp::shutdown();
}
