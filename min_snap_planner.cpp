#include "rclcpp/rclcpp.hpp"
#include "nav_msgs/msg/path.hpp"
#include "nav_msgs/msg/occupancy_grid.hpp"
#include "trajectory_msgs/msg/multi_dof_joint_trajectory.hpp"//带加速度的path
#include "geometry_msgs/msg/pose_stamped.hpp"//path的元素
#include "geometry_msgs/msg/transform.hpp"//traj的元素
#include "geometry_msgs/msg/twist.hpp"//存放加速度和速度

#include <Eigen/Dense>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <numeric>
#include <queue>
#include <string>
#include <utility>
#include <vector>

namespace
{

// 多项式阶数与系数个数：7 阶多项式可保证 snap（4 阶导）分段连续
constexpr int kOrder = 7;                 // 7th-order polynomial
constexpr int kCoeffs = kOrder + 1;       // 每段多项式 8 个系数
using Vec8 = Eigen::Matrix<double, kCoeffs, 1>;   // 单段单轴的系数向量
using Row8 = Eigen::Matrix<double, 1, kCoeffs>;   // 系数行向量（用于构造约束行）

// 平面点 / 二维向量
struct Vec2
{
  double x = 0.0;
  double y = 0.0;
};

// Vec2 的线性代数小工具：加减、数乘、点积、叉积、模长
Vec2 operator+(const Vec2 & a, const Vec2 & b) {return Vec2{a.x + b.x, a.y + b.y};}
Vec2 operator-(const Vec2 & a, const Vec2 & b) {return Vec2{a.x - b.x, a.y - b.y};}
Vec2 operator*(const Vec2 & a, double s) {return Vec2{a.x * s, a.y * s};}
double dot(const Vec2 & a, const Vec2 & b) {return a.x * b.x + a.y * b.y;}
double cross(const Vec2 & a, const Vec2 & b) {return a.x * b.y - a.y * b.x;}
double norm(const Vec2 & a) {return std::hypot(a.x, a.y);}

//栅格地图
struct GridMap
{
  std::vector<int8_t> data;      
  int width = 0;                 
  int height = 0;               
  double resolution = 0.05;     
  double origin_x = 0.0;         
  double origin_y = 0.0;         
  std::string frame_id = "map";  
  bool ready = false;            //是否已收到地图
};

// 连接点的矩形可行域 [lx, ux] x [ly, uy]，用于把轨迹限制在安全区域内
struct Box
{
  double lx = 0.0;
  double ux = 0.0;
  double ly = 0.0;
  double uy = 0.0;
};

// 一段轨迹：时长 T 与 x/y 两个方向的多项式系数（归一化时间 tau = t / T）
struct Segment
{
  double T = 1.0;
  Vec8 cx = Vec8::Zero();
  Vec8 cy = Vec8::Zero();
};

// 线性不等式约束集合 A x <= b（每行一个约束 + 对应右端项）
struct IneqSet
{
  std::vector<Eigen::RowVectorXd> rows;
  std::vector<double> b;
};


// 判断栅格 (gx, gy) 是否被占据；越界一律按占据处理
bool gridCellOccupied(const GridMap & g, int gx, int gy, int occupied_thresh)
{
  if (gx < 0 || gy < 0 || gx >= g.width || gy >= g.height) 
  {
    return true;
  }
  const int v = g.data[static_cast<size_t>(gy) * g.width + gx];
  return v >= occupied_thresh;
}

// 判断世界坐标 (x, y) 处是否可通行。
// margin == 0: 只检查该点所在栅格。
// margin > 0 : 检查以该点为圆心、margin 为半径的圆内所有栅格（用于考虑机器人半径）。
bool isFreeWorld(const GridMap & g, double x, double y, double margin, int occupied_thresh)
{
  if (g.width <= 0 || g.height <= 0 || g.resolution <= 0.0) 
  {
    return false;
  }
  // 世界坐标 -> 栅格索引
  const int gx = static_cast<int>(std::floor((x - g.origin_x) / g.resolution));
  const int gy = static_cast<int>(std::floor((y - g.origin_y) / g.resolution));
  if (gx < 0 || gy < 0 || gx >= g.width || gy >= g.height) 
  {
    return false;
  }
  if (margin <= 1e-9) {
    return !gridCellOccupied(g, gx, gy, occupied_thresh);
  }

  // 枚举圆盘覆盖到的栅格，按格心到圆心的距离筛选
  const int radius = static_cast<int>(std::ceil(margin / g.resolution));
  for (int dy = -radius; dy <= radius; ++dy) 
  {
    for (int dx = -radius; dx <= radius; ++dx) 
    {
      const double dist = std::hypot(static_cast<double>(dx), static_cast<double>(dy)) *
        g.resolution;
      if (dist > margin + 1e-9) 
      {
        continue;
      }
      if (gridCellOccupied(g, gx + dx, gy + dy, occupied_thresh)) 
      {
        return false;
      }
    }
  }
  return true;
}

// 沿线段 a->b 按半分辨率步长采样，判断整条线段是否无碰撞
bool segmentFree(
  const GridMap & g, const Vec2 & a, const Vec2 & b, double margin, int occupied_thresh)
{
  const double len = norm(b - a);
  const int samples = std::max(2, static_cast<int>(std::ceil(len / (g.resolution * 0.5))));
  for (int i = 0; i <= samples; ++i) 
  {
    const double t = static_cast<double>(i) / static_cast<double>(samples);
    const Vec2 p = a + (b - a) * t;
    if (!isFreeWorld(g, p.x, p.y, margin, occupied_thresh)) 
    {
      return false;
    }
  }
  return true;
}

// --------------------------------------------------------------------------
// 膨胀地图与膨胀地图上的参考路径
// --------------------------------------------------------------------------

// 对栅格地图做圆形膨胀：距离障碍小于 radius 的栅格都按占据处理。
// 这里只用于轨迹优化，不改变 /map，也不改变 A* 的输出。
GridMap inflateMap(const GridMap & src, double radius)
{
  GridMap dst = src;
  dst.data.assign(src.data.size(), 0);
  if (radius <= 0.0 || src.width <= 0 || src.height <= 0 || src.resolution <= 0.0) 
  {
    dst.data = src.data;
    return dst;
  }

  const int r = static_cast<int>(std::ceil(radius / src.resolution));
  // 预计算圆形膨胀核：只保留与中心距离 <= radius 的偏移，
  // 这样膨胀出来的是圆盘而不是正方形，更接近机器人的实际外形。
  std::vector<std::pair<int, int>> offsets;
  offsets.reserve(static_cast<size_t>((2 * r + 1) * (2 * r + 1)));
  for (int dy = -r; dy <= r; ++dy) 
  {
    for (int dx = -r; dx <= r; ++dx) 
    {
      const double d = std::hypot(static_cast<double>(dx), static_cast<double>(dy)) *
        src.resolution;
      if (d <= radius + 1e-9) 
      {
        offsets.emplace_back(dx, dy);
      }
    }
  }

  // 对每个原始障碍栅格，把膨胀核覆盖到的栅格都标成占据。
  // 这里阈值 50 与 occupied_thresh 默认值一致：地图值为 0（空闲）/ 100（占据）。
  for (int y = 0; y < src.height; ++y) 
  {
    for (int x = 0; x < src.width; ++x) 
    {
      if (src.data[static_cast<size_t>(y) * src.width + x] < 50) 
      {
        continue;
      }
      for (const auto & off : offsets) 
      {
        const int nx = x + off.first;
        const int ny = y + off.second;
        if (nx >= 0 && ny >= 0 && nx < src.width && ny < src.height) 
        {
          dst.data[static_cast<size_t>(ny) * src.width + nx] = 100;
        }
      }
    }
  }
  dst.ready = src.ready;
  return dst;
}

// 每个栅格到最近障碍的欧氏距离（米），用于在膨胀地图上优先走通道中间。
std::vector<float> buildClearance(const GridMap & g)
{
  const size_t n = g.data.size();
  std::vector<float> dist(n, std::numeric_limits<float>::infinity());
  if (g.width <= 0 || g.height <= 0 || g.resolution <= 0.0) 
  {
    return dist;
  }

  using Item = std::pair<float, int>;
  std::priority_queue<Item, std::vector<Item>, std::greater<Item>> open;
  // 多源 Dijkstra：把所有障碍栅格作为源点，向自由空间传播欧氏距离。
  // 得到每个自由栅格“到最近障碍的距离”，供参考路径的居中代价使用。
  for (size_t i = 0; i < n; ++i) 
  {
    if (g.data[i] >= 50) 
    {
      dist[i] = 0.0f;
      open.emplace(0.0f, static_cast<int>(i));
    }
  }

  static const int dx[8] = {1, -1, 0, 0, 1, 1, -1, -1};
  static const int dy[8] = {0, 0, 1, -1, 1, -1, 1, -1};
  while (!open.empty()) 
  {
    const auto item = open.top();
    open.pop();
    const float d = item.first;
    const int idx = item.second;
    if (d > dist[static_cast<size_t>(idx)]) 
    {
      continue;
    }
    const int x = idx % g.width;
    const int y = idx / g.width;
    for (int k = 0; k < 8; ++k) 
    {
      const int nx = x + dx[k];
      const int ny = y + dy[k];
      if (nx < 0 || ny < 0 || nx >= g.width || ny >= g.height) 
      {
        continue;
      }
      const double step = (dx[k] != 0 && dy[k] != 0) ?
        g.resolution * std::sqrt(2.0) : g.resolution;
      const int ni = ny * g.width + nx;
      const float nd = d + static_cast<float>(step);
      if (nd < dist[static_cast<size_t>(ni)]) 
      {
        dist[static_cast<size_t>(ni)] = nd;
        open.emplace(nd, ni);
      }
    }
  }
  return dist;
}

// 如果给定栅格落在膨胀层里，向外找最近的空闲栅格（BFS）。
bool nearestFreeCell(const GridMap & g, int & cx, int & cy, int occupied_thresh)
{
  if (g.width <= 0 || g.height <= 0) 
  {
    return false;
  }
  const auto inside = [&](int x, int y) 
    {
      return x >= 0 && y >= 0 && x < g.width && y < g.height;
    };
  const auto freeAt = [&](int x, int y) 
    {
      return g.data[static_cast<size_t>(y) * g.width + x] < occupied_thresh;
    };

  cx = std::clamp(cx, 0, g.width - 1);
  cy = std::clamp(cy, 0, g.height - 1);
  if (freeAt(cx, cy)) 
  {
    return true;
  }

  // 起终点或参考点可能落在膨胀层里，用 4 邻域 BFS 向外找最近的可通行栅格。
  // BFS 允许穿过膨胀层，只看最终找到的空闲栅格。
  std::vector<char> visited(static_cast<size_t>(g.width) * g.height, 0);
  std::queue<std::pair<int, int>> q;
  q.emplace(cx, cy);
  visited[static_cast<size_t>(cy) * g.width + cx] = 1;
  static const int dx[4] = {1, -1, 0, 0};
  static const int dy[4] = {0, 0, 1, -1};
  while (!q.empty()) 
  {
    const auto cur = q.front();
    q.pop();
    for (int k = 0; k < 4; ++k) 
    {
      const int nx = cur.first + dx[k];
      const int ny = cur.second + dy[k];
      if (!inside(nx, ny) || visited[static_cast<size_t>(ny) * g.width + nx]) 
      {
        continue;
      }
      if (freeAt(nx, ny)) 
      {
        cx = nx;
        cy = ny;
        return true;
      }
      visited[static_cast<size_t>(ny) * g.width + nx] = 1;
      q.emplace(nx, ny);
    }
  }
  return false;
}

// 在膨胀地图上做一次带间隙代价的 A*，生成避开膨胀区的参考折线。
// 这不是修改 astar_planner_demo，只是轨迹优化内部的参考路径修复。
std::vector<Vec2> clearanceAstar(
  const GridMap & g, const std::vector<float> & clearance, const Vec2 & start,
  const Vec2 & goal, int occupied_thresh, double clearance_weight,
  double preferred_clearance)
{
  std::vector<Vec2> empty;
  if (g.width <= 0 || g.height <= 0 || g.resolution <= 0.0) 
  {
    return empty;
  }
  const size_t cell_count = static_cast<size_t>(g.width) * g.height;
  if (clearance.size() != cell_count) 
  {
    return empty;
  }

  auto toCell = [&](const Vec2 & p, int & x, int & y) 
    {
      x = static_cast<int>(std::floor((p.x - g.origin_x) / g.resolution));
      y = static_cast<int>(std::floor((p.y - g.origin_y) / g.resolution));
    };
  int sx = 0;
  int sy = 0;
  int gx = 0;
  int gy = 0;
  toCell(start, sx, sy);
  toCell(goal, gx, gy);
  if (!nearestFreeCell(g, sx, sy, occupied_thresh) ||
    !nearestFreeCell(g, gx, gy, occupied_thresh))
  {
    return empty;
  }

  const int w = g.width;
  const int n = w * g.height;
  const int si = sy * w + sx;
  const int gi = gy * w + gx;
  const double inf = std::numeric_limits<double>::infinity();

  auto freeCell = [&](int x, int y) 
    {
      return x >= 0 && y >= 0 && x < w && y < g.height &&
             g.data[static_cast<size_t>(y) * w + x] < occupied_thresh;
    };
  auto heuristic = [&](int x, int y) 
    {
      return std::hypot(static_cast<double>(x - gx), static_cast<double>(y - gy)) *
             g.resolution;
    };

  std::vector<double> gscore(static_cast<size_t>(n), inf);
  std::vector<int> parent(static_cast<size_t>(n), -1);
  std::vector<char> closed(static_cast<size_t>(n), 0);
  using Item = std::pair<double, int>;
  std::priority_queue<Item, std::vector<Item>, std::greater<Item>> open;
  gscore[static_cast<size_t>(si)] = 0.0;
  open.emplace(heuristic(sx, sy), si);

  static const int dx[8] = {1, -1, 0, 0, 1, 1, -1, -1};
  static const int dy[8] = {0, 0, 1, -1, 1, -1, 1, -1};

  // 搜索代价 = 实际步长 + 间隙惩罚。
  // 间隙惩罚让路径尽量走通道中间，而不是贴着膨胀层边缘走；
  // 斜向移动时检查两个正交方向，避免从障碍角上切过去。
  while (!open.empty()) 
  {
    const int cur = open.top().second;
    open.pop();
    if (closed[static_cast<size_t>(cur)]) 
    {
      continue;
    }
    closed[static_cast<size_t>(cur)] = 1;
    if (cur == gi) 
    {
      break;
    }
    const int cx = cur % w;
    const int cy = cur / w;
    for (int k = 0; k < 8; ++k) 
    {
      const int nx = cx + dx[k];
      const int ny = cy + dy[k];
      if (!freeCell(nx, ny)) 
      {
        continue;
      }
      if (dx[k] != 0 && dy[k] != 0 &&
        (!freeCell(cx + dx[k], cy) || !freeCell(cx, cy + dy[k])))
      {
        continue;
      }
      const int ni = ny * w + nx;
      if (closed[static_cast<size_t>(ni)]) 
      {
        continue;
      }
      const double step = (dx[k] != 0 && dy[k] != 0) ?
        g.resolution * std::sqrt(2.0) : g.resolution;
      const double d = static_cast<double>(clearance[static_cast<size_t>(ni)]);
      const double penalty = clearance_weight * std::max(0.0, preferred_clearance - d);
      const double ng = gscore[static_cast<size_t>(cur)] + step + penalty;
      if (ng < gscore[static_cast<size_t>(ni)]) 
      {
        gscore[static_cast<size_t>(ni)] = ng;
        parent[static_cast<size_t>(ni)] = cur;
        open.emplace(ng + heuristic(nx, ny), ni);
      }
    }
  }

  if (gi != si && parent[static_cast<size_t>(gi)] < 0) 
  {
    return empty;
  }

  std::vector<Vec2> path;
  for (int cur = gi; cur >= 0; cur = parent[static_cast<size_t>(cur)]) 
  {
    const int x = cur % w;
    const int y = cur / w;
    path.push_back(Vec2{
      g.origin_x + (static_cast<double>(x) + 0.5) * g.resolution,
      g.origin_y + (static_cast<double>(y) + 0.5) * g.resolution});
    if (cur == si) 
    {
      break;
    }
  }
  std::reverse(path.begin(), path.end());
  return path;
}

// 如果原 A* 路径已经满足膨胀层，就直接用它；否则在膨胀地图上重找一条参考路径。
// 注意：这只是轨迹优化内部的参考路径修复，A* 包和 /path 都不变。
std::vector<Vec2> makeReferencePath(
  const std::vector<Vec2> & raw, const GridMap & inflated,
  const std::vector<float> & clearance, int occupied_thresh,
  double clearance_weight, double preferred_clearance, bool & rerouted)
{
  rerouted = false;
  if (raw.size() < 2) 
  {
    return raw;
  }

  // 逐段检查原 A* 折线是否穿过膨胀层：
  // 只要有一段不满足，就认为整条路径需要重找。
  bool valid = true;
  for (size_t i = 0; i + 1 < raw.size() && valid; ++i) 
  {
    if (!segmentFree(inflated, raw[i], raw[i + 1], 0.0, occupied_thresh)) 
    {
      valid = false;
    }
  }
  if (valid) 
  {
    // 原路径已经离障碍足够远，直接沿用，避免不必要的绕行
    return raw;
  }

  // 原路径贴边/穿膨胀层：在膨胀地图上重新搜索一条居中参考路径
  auto alternative = clearanceAstar(
    inflated, clearance, raw.front(), raw.back(), occupied_thresh,
    clearance_weight, preferred_clearance);
  if (alternative.size() < 2) 
  {
    // 重找失败（例如膨胀后目标不可达）时退回原路径，
    // 后续由 QP 的段内采样约束尽量避开障碍
    return raw;
  }
  rerouted = true;
  return alternative;
}

// --------------------------------------------------------------------------
// 连接点（waypoint）提取与简化
// --------------------------------------------------------------------------

// 从 A* 原始折线路径中提取用于优化的连接点：
//   1) 去掉重复点；
//   2) 去掉共线冗余点，只保留拐角；
//   3) 视线简化：能直线直达就跳到最远可见点（减少段数）；
//   4) 对过长的直线段按 insert_spacing 等距插点，避免单段跨度过大。
std::vector<Vec2> extractWaypoints(
  const std::vector<Vec2> & raw, const GridMap & g, int occupied_thresh, double insert_spacing, double obstacle_margin)
{
  // 1) 相邻重复点去重
  std::vector<Vec2> pts;
  pts.reserve(raw.size());
  for (const auto & p : raw) 
  {
    if (pts.empty() || norm(p - pts.back()) > 1e-6) 
    {
      pts.push_back(p);
    }
  }

  // 2) 删除共线且同向的中间点（叉积近似为 0 且点积大于 0）
  bool changed = true;
  while (changed && pts.size() > 2) 
  {
    changed = false;
    for (size_t i = 1; i + 1 < pts.size(); ++i) 
    {
      const Vec2 a = pts[i] - pts[i - 1];
      const Vec2 b = pts[i + 1] - pts[i];
      const double scale = std::max(1.0, norm(a) * norm(b));
      if (std::fabs(cross(a, b)) <= 1e-9 * scale && dot(a, b) >= 0.0) 
      {
        pts.erase(pts.begin() + static_cast<long>(i));
        changed = true;
        break;
      }
    }
  }

  if (pts.size() < 2) 
  {
    return pts;
  }

  // 3) 贪心视线简化：从当前点出发，直接连到最远的无碰撞点
  std::vector<Vec2> simple;
  simple.push_back(pts.front());
  size_t i = 0;
  while (i + 1 < pts.size()) 
  {
    size_t j = pts.size() - 1;
    while (j > i + 1 && !segmentFree(g, pts[i], pts[j], obstacle_margin, occupied_thresh))    //0.0->obsta
    {
      --j;
    }
    simple.push_back(pts[j]);
    i = j;
  }

  // 4) 长直线段按约 insert_spacing 米等距插入连接点
  if (insert_spacing <= 0.0) 
  {
    return simple;
  }
  std::vector<Vec2> result;
  result.push_back(simple.front());
  for (size_t k = 0; k + 1 < simple.size(); ++k) 
  {
    const Vec2 d = simple[k + 1] - simple[k];
    const double len = norm(d);
    const int pieces = std::max(1, static_cast<int>(std::ceil(len / insert_spacing)));
    for (int p = 1; p < pieces; ++p) 
    {
      result.push_back(simple[k] + d * (static_cast<double>(p) / static_cast<double>(pieces)));
    }
    result.push_back(simple[k + 1]);
  }
  return result;
}

// --------------------------------------------------------------------------
// Minimum Snap 二次规划（QP）：代价矩阵、约束与求解器
// --------------------------------------------------------------------------

// 构造 d 阶导数对系数向量的"求导行向量"：
//   d^d/dtau^d (sum_i c_i tau^i) = sum_{i>=d} c_i * i!/(i-d)! * tau^(i-d)
// 归一化时间 tau = t / T。
Row8 derivRow(int d, double tau)
{
  Row8 r = Row8::Zero();
  for (int i = d; i < kCoeffs; ++i) 
  {
    double factor = 1.0;
    for (int k = 0; k < d; ++k) 
    {
      factor *= static_cast<double>(i - k);
    }
    r[i] = factor * std::pow(tau, i - d);
  }
  return r;
}

// 归一化时间下 snap（4 阶导）平方积分的 Hessian：
//   H(i,j) = 2 * i!/(i-4)! * j!/(j-4)! / (i+j-7)
// 真实时间下代价为 (1/T^7) * tau 空间代价，故使用时需除以 T^7。
const Eigen::Matrix<double, kCoeffs, kCoeffs> & snapUnitHessian()
{
  static const Eigen::Matrix<double, kCoeffs, kCoeffs> q = []
  {
      Eigen::Matrix<double, kCoeffs, kCoeffs> m =
        Eigen::Matrix<double, kCoeffs, kCoeffs>::Zero();
      for (int i = 4; i < kCoeffs; ++i) 
      {
        for (int j = 4; j < kCoeffs; ++j) 
        {
          double ci = 1.0;
          double cj = 1.0;
          for (int k = 0; k < 4; ++k) 
          {
            ci *= static_cast<double>(i - k);
            cj *= static_cast<double>(j - k);
          }
          m(i, j) = 2.0 * ci * cj / static_cast<double>(i + j - 7);
        }
      }
      return m;
  }();
  return q;
}

// 在归一化时刻 tau 处计算某段多项式的 d 阶导数。
// 由于对 tau 求导会引入 1/T^d，这里除以 T^d 还原成真实时间导数。
double evalPoly(const Vec8 & c, int derivative, double tau, double T)
{
  //行向量与系数向量做点积
  return derivRow(derivative, tau).dot(c) / std::pow(T, derivative);
}

// 组装 QP 等式约束 A * [cx; cy] = [bx; by]（x、y 两轴共用同一个 A）。
// 约束包括：
//   1) 起点固定（位置 + 速度/加速度/jerk 为 0）；
//   2) 终点固定（位置 + 速度/加速度/jerk 为 0）；
//   3) 内部分段衔接处位置、速度、加速度、jerk 连续；
//   4) 因空间过窄被"固定"的连接点位置取路径点坐标；
//   5) 需要减速停止的连接点加 v = a = j = 0 约束。
void buildEquality(
  const std::vector<double> & T, const std::vector<Vec2> & wp,
  const std::vector<bool> & fixed, const std::vector<bool> & stop,
  Eigen::MatrixXd & A, Eigen::VectorXd & bx, Eigen::VectorXd & by)
{
  const int N = static_cast<int>(T.size());   // 轨迹段数
  const int n = 8 * N;                        // 决策变量总数（每段 8 个系数，x/y 各解一次）
  std::vector<Eigen::RowVectorXd> rows;
  std::vector<double> rhs_x;
  std::vector<double> rhs_y;

  // 小工具：收集一行约束及其 x/y 右端项
  auto push = [&](const Eigen::RowVectorXd & row, double x, double y) 
  {
      rows.push_back(row);
      rhs_x.push_back(x);
      rhs_y.push_back(y);
    };

  // 约束 1)：固定起点位置，且起始 v = a = j = 0（静止起飞）
  for (int d = 0; d <= 3; ++d) 
  {
    Eigen::RowVectorXd row = Eigen::RowVectorXd::Zero(n);
    row.segment<kCoeffs>(0) = derivRow(d, 0.0) / std::pow(T.front(), d);
    push(row, d == 0 ? wp.front().x : 0.0, d == 0 ? wp.front().y : 0.0);
  }

  // 约束 2)：固定终点位置，且末端 v = a = j = 0（静止到达）
  const int last = N - 1;
  for (int d = 0; d <= 3; ++d) 
  {
    Eigen::RowVectorXd row = Eigen::RowVectorXd::Zero(n);
    row.segment<kCoeffs>(8 * last) = derivRow(d, 1.0) / std::pow(T.back(), d);
    push(row, d == 0 ? wp.back().x : 0.0, d == 0 ? wp.back().y : 0.0);
  }

  // 约束 3)：相邻两段在衔接点（上一段 tau=1、下一段 tau=0）各阶导数相等。
  // d 取 0~4：位置、速度、加速度、jerk、snap 全部连续。
  // 其中 d <= 3 保证运动学连续，d == 4 让 snap 也连续，曲线更顺、不会在连接点出现“顿挫”。
  for (int k = 1; k < N; ++k) 
  {
    for (int d = 0; d <= 4; ++d) 
    {
      Eigen::RowVectorXd row = Eigen::RowVectorXd::Zero(n);
      row.segment<kCoeffs>(8 * (k - 1)) =
        derivRow(d, 1.0) / std::pow(T[k - 1], d);
      row.segment<kCoeffs>(8 * k) =
        -derivRow(d, 0.0) / std::pow(T[k], d);
      push(row, 0.0, 0.0);
    }
  }

  // 约束 4)：空间过窄的连接点直接固定到原路径点上（等价于矩形收缩成一点）
  for (int k = 1; k < N; ++k) 
  {
    if (!fixed[k]) 
    {
      continue;
    }
    Eigen::RowVectorXd row = Eigen::RowVectorXd::Zero(n);
    row[8 * k] = 1.0;
    push(row, wp[k].x, wp[k].y);
  }

  // 约束 5)：指定连接点处速度为 0（v = a = j = 0），便于原地绕行/停顿
  for (int k = 1; k < N; ++k) 
  {
    if (!stop[k]) 
    {
      continue;
    }
    for (int d = 1; d <= 3; ++d) 
    {
      Eigen::RowVectorXd row = Eigen::RowVectorXd::Zero(n);
      row.segment<kCoeffs>(8 * k) = derivRow(d, 0.0) / std::pow(T[k], d);
      push(row, 0.0, 0.0);
    }
  }

  // 把收集到的约束行整理成矩阵形式
  const int m = static_cast<int>(rows.size());
  A.resize(m, n);
  bx.resize(m);
  by.resize(m);
  for (int r = 0; r < m; ++r)
  {
    A.row(r) = rows[static_cast<size_t>(r)];
    bx[r] = rhs_x[static_cast<size_t>(r)];
    by[r] = rhs_y[static_cast<size_t>(r)];
  }
}

// 把矩形可行域转成不等式约束：
//   1) 每个连接点的位置落在其矩形内；
//   2) 每段在 tau = 0.25/0.5/0.75 处的位置也落在对应采样点的矩形内，
//      防止多项式在连接点之间鼓出膨胀层。
// fixed 的连接点已由等式约束固定，不再生成不等式。
void buildBoxInequalities(
  int N, const std::vector<Box> & boxes, const std::vector<bool> & fixed,
  const std::vector<std::vector<Box>> & sample_boxes, IneqSet & ix, IneqSet & iy)
{
  const int n = 8 * N;
  for (int k = 1; k < N; ++k) 
  {
    if (fixed[k]) 
    {
      continue;
    }
    // 取出第 k 段起点的零阶系数 c_{k,0}，即该连接点的位置
    Eigen::RowVectorXd a = Eigen::RowVectorXd::Zero(n);
    a[8 * k] = 1.0;

    ix.rows.push_back(a);
    ix.b.push_back(boxes[k].ux);
    ix.rows.push_back(-a);
    ix.b.push_back(-boxes[k].lx);

    iy.rows.push_back(a);
    iy.b.push_back(boxes[k].uy);
    iy.rows.push_back(-a);
    iy.b.push_back(-boxes[k].ly);
  }

  // 段内采样约束：只在连接点处约束位置还不够，
  // 多项式可能在两个连接点之间“鼓”进障碍或膨胀层。
  // 这里在每段的 tau = 0.25 / 0.5 / 0.75 处再加一组位置盒子约束，
  // 把整段曲线压在参考折线附近的安全区域内。
  // 采样盒子的行向量是 tau 处的零阶导数列，等价于取该点位置。
  static const double taus[3] = {0.25, 0.5, 0.75};
  for (int s = 0; s < N; ++s) 
  {
    if (sample_boxes.size() < static_cast<size_t>(N)) 
    {
      break;
    }
    const auto & seg_boxes = sample_boxes[static_cast<size_t>(s)];
    if (seg_boxes.size() < 3) 
    {
      continue;
    }
    for (int j = 0; j < 3; ++j) 
    {
      const Box & b = seg_boxes[static_cast<size_t>(j)];
      Eigen::RowVectorXd a = Eigen::RowVectorXd::Zero(n);
      // derivRow(0, tau) 即 tau 处的位置表达式，乘上本段系数即采样点坐标
      a.segment<kCoeffs>(8 * s) = derivRow(0, taus[j]);

      ix.rows.push_back(a);
      ix.b.push_back(b.ux);
      ix.rows.push_back(-a);
      ix.b.push_back(-b.lx);

      iy.rows.push_back(a);
      iy.b.push_back(b.uy);
      iy.rows.push_back(-a);
      iy.b.push_back(-b.ly);
    }
  }
}

// 求解等式约束 QP：min 0.5 x^T H x  s.t. A x = b
// 通过对 KKT 矩阵 [H A^T; A 0] 做 QR 分解直接求解；残差过大则认为数值失败。
bool solveEqualityQP(
  const Eigen::MatrixXd & H, const Eigen::MatrixXd & A, const Eigen::VectorXd & b,
  Eigen::VectorXd & x, Eigen::VectorXd & lambda)
{
  const int n = static_cast<int>(H.rows());
  const int m = static_cast<int>(A.rows());
  Eigen::MatrixXd K = Eigen::MatrixXd::Zero(n + m, n + m);//大零矩阵
  K.topLeftCorner(n, n) = H;				//左上块赋值
  Eigen::VectorXd rhs = Eigen::VectorXd::Zero(n + m);
  if (m > 0) 
  {
    K.topRightCorner(n, m) = A.transpose();//转置		//右上块 = A^T
    K.bottomLeftCorner(m, n) = A;			//左下块 = A
    rhs.tail(m) = b;					//尾部块
  }

  Eigen::ColPivHouseholderQR<Eigen::MatrixXd> qr(K);	//QR分解
  const Eigen::VectorXd sol = qr.solve(rhs);		//求解线性方程
  const double residual = (K * sol - rhs).norm();	//矩阵向量乘+范数
  if (!std::isfinite(residual) || residual > 1e-5 * (1.0 + rhs.norm())) 
  {
    return false;
  }
  x = sol.head(n);					//头部块
  lambda = (m > 0) ? sol.tail(m) : Eigen::VectorXd();  // 尾部的乘子供有效集判断
  return true;
}

// 求解带矩形不等式约束的 QP：min 0.5 x^T H x  s.t. Aeq x = beq, Aineq x <= bineq
// 采用有效集法（active-set）：
//   1) 先只带等式约束求解；
//   2) 若存在违反的不等式，把最严重的一行加入有效集再解；
//   3) 若所有不等式都满足，检查有效集约束的 Lagrange 乘子，剔除最负的一行；
//   4) 直到可行且乘子非负，得到最优解。
bool solveBoxQP(
  const Eigen::MatrixXd & H, const Eigen::MatrixXd & Aeq, const Eigen::VectorXd & beq,
  const IneqSet & ineq, Eigen::VectorXd & xout)
{
  const int n = static_cast<int>(H.rows());
  const int meq = static_cast<int>(Aeq.rows());
  const int nineq = static_cast<int>(ineq.rows.size());

  if (nineq == 0) 
  {
    Eigen::VectorXd lambda;
    return solveEqualityQP(H, Aeq, beq, xout, lambda);
  }

  // 把 IneqSet 转成矩阵形式，方便按行索引
  Eigen::MatrixXd Aineq(nineq, n);		//稠密矩阵
  Eigen::VectorXd bineq(nineq);
  for (int i = 0; i < nineq; ++i) 
  {
    Aineq.row(i) = ineq.rows[static_cast<size_t>(i)];
    bineq[i] = ineq.b[static_cast<size_t>(i)];
  }

  std::vector<int> active;   // 当前有效（取等号）的不等式行下标
  for (int iter = 0; iter < 300; ++iter) 
  {
    // 组装「等式约束 + 当前有效集」并求解
    const int m = meq + static_cast<int>(active.size());
    Eigen::MatrixXd A(m, n);
    Eigen::VectorXd b(m);
    A.topRows(meq) = Aeq;		//顶部行放Aeq
    b.head(meq) = beq;			//向量头部
    for (size_t j = 0; j < active.size(); ++j) 
    {
      //A.row()行视图，可读可写
      A.row(meq + static_cast<int>(j)) = Aineq.row(active[j]);//按索引拷贝某一行
      b[meq + static_cast<int>(j)] = bineq[active[j]];
    }

    Eigen::VectorXd x;
    Eigen::VectorXd lambda;
    if (!solveEqualityQP(H, A, b, x, lambda)) 
    {
      return false;
    }

    // 检查是否有未加入有效集但被违反的约束，取违反量最大者加入
    int worst = -1;
    double worst_violation = 1e-6;
    for (int i = 0; i < nineq; ++i) 
    {
      if (std::find(active.begin(), active.end(), i) != active.end()) 
      {
        continue;
      }
      //行向量与列向量点积
      const double violation = Aineq.row(i).dot(x) - bineq[i];
      if (violation > worst_violation) 
      {
        worst_violation = violation;
        worst = i;
      }
    }
    if (worst >= 0) {
      active.push_back(worst);
      continue;
    }

    // 全部可行：检查有效集乘子，若为负说明该约束不该生效，剔除后重解
    int drop = -1;
    double worst_mu = -1e-6;
    for (size_t j = 0; j < active.size(); ++j) 
    {
      const double mu = lambda[meq + static_cast<int>(j)];
      if (mu < worst_mu) 
      {
        worst_mu = mu;
        drop = static_cast<int>(j);
      }
    }
    if (drop >= 0) 
    {
      active.erase(active.begin() + drop);
      continue;
    }

    xout = x;
    return true;
  }
  return false;   // 迭代次数用尽仍未收敛
}

// --------------------------------------------------------------------------
// ROS 节点
// --------------------------------------------------------------------------

}  // namespace

// Minimum Snap 规划节点：
//   订阅 /map 与 /path，规划完成后发布 /min_snap_path 和 /trajectory。
class MinSnapPlanner : public rclcpp::Node
{
public:
  MinSnapPlanner() : Node("min_snap")
  {
    // ---------------- 参数读取 ----------------
    v_ref_ = declare_parameter<double>("v_ref", 1.0);                 // 参考速度，用于分配每段时间
    min_seg_time_ = declare_parameter<double>("min_seg_time", 1.0);   // 单段最短时长 (s)
    max_seg_time_ = declare_parameter<double>("max_seg_time", 4.0);   // 单段最长时长 (s)
    insert_spacing_ = declare_parameter<double>("insert_spacing", 1.0); // 长直线插点间距 (m)
    max_rect_half_ = declare_parameter<double>("max_rect_half", 0.50); // 矩形可行域最大半边长 (m)
    min_rect_half_ = declare_parameter<double>("min_rect_half", 0.01); // 矩形可行域最小半边长 (m)
    obstacle_margin_ = declare_parameter<double>("obstacle_margin", 0.05); // 碰撞检测安全裕度 (m)
    sample_dt_ = declare_parameter<double>("sample_dt", 0.02);         // 轨迹发布采样周期 (s)
    collision_dt_ = declare_parameter<double>("collision_dt", 0.01);   // 碰撞检测采样周期 (s)
    max_iterations_ = declare_parameter<int>("max_iters", 20);    // 收紧矩形最大迭代次数
    occupied_thresh_ = declare_parameter<int>("occupied_thresh", 50);  // 占据判定阈值
    inflation_radius_ = declare_parameter<double>("inflation_radius", 0.35); // 轨迹优化膨胀半径 (m)
    clearance_weight_ = declare_parameter<double>("clearance_weight", 0.5);  // 膨胀参考路径的居中权重
    auto_reroute_ = declare_parameter<bool>("auto_reroute", true);          // 穿过膨胀区时是否重找参考路径

    // 参数合法性保护，避免除零或时间区间反置
    if (v_ref_ <= 0.0) 
    {
      v_ref_ = 1.0;
    }
    if (inflation_radius_ < 0.0) 
    {
      inflation_radius_ = 0.0;
    }
    if (clearance_weight_ < 0.0) 
    {
      clearance_weight_ = 0.0;
    }
    if (min_seg_time_ <= 0.0) 
    {
      min_seg_time_ = 0.5;
    }
    if (max_seg_time_ < min_seg_time_) 
    {
      max_seg_time_ = min_seg_time_;
    }
    if (sample_dt_ <= 0.0)
    {
      sample_dt_ = 0.02;
    }
    if (collision_dt_ <= 0.0) 
    {
      collision_dt_ = 0.01;
    }

    // ---------------- 通信接口 ----------------
    // 地图与路径都使用 transient_local，保证晚启动的节点也能收到最后一帧
    const auto map_qos = rclcpp::QoS(1).transient_local();
    map_sub_ = create_subscription<nav_msgs::msg::OccupancyGrid>(
      "/map", map_qos,
      [this](const nav_msgs::msg::OccupancyGrid::SharedPtr msg) {onMap(msg);});

    const auto path_qos = rclcpp::QoS(1).transient_local();
    path_sub_ = create_subscription<nav_msgs::msg::Path>(
      "/path", path_qos,
      [this](const nav_msgs::msg::Path::SharedPtr msg) {onPath(msg);});

    pub_path_ = create_publisher<nav_msgs::msg::Path>("/min_snap_path", path_qos);
    pub_traj_ = create_publisher<trajectory_msgs::msg::MultiDOFJointTrajectory>(
      "/trajectory", path_qos);

    RCLCPP_INFO(
      get_logger(),
      "Minimum snap 就绪: v_ref=%.2f m/s, 每段 %.2f~%.2f s, 矩形半边长 %.2f~%.2f m, 膨胀 %.2f m",
      v_ref_, min_seg_time_, max_seg_time_, min_rect_half_, max_rect_half_, inflation_radius_);
  }

private:
  // /map 回调：缓存地图信息，若已有路径则立即重新规划
  void onMap(const nav_msgs::msg::OccupancyGrid::SharedPtr msg)
  {
    grid_.width = static_cast<int>(msg->info.width);
    grid_.height = static_cast<int>(msg->info.height);
    grid_.resolution = msg->info.resolution;
    grid_.origin_x = msg->info.origin.position.x;
    grid_.origin_y = msg->info.origin.position.y;
    grid_.frame_id = msg->header.frame_id.empty() ? "map" : msg->header.frame_id;
    grid_.data = msg->data;
    grid_.ready = true;

    // 轨迹优化专用膨胀地图：不改变 /map，也不影响 A* 包
    inflated_grid_ = inflateMap(grid_, inflation_radius_);
    inflated_grid_.frame_id = grid_.frame_id;
    inflated_grid_.ready = true;
    clearance_ = buildClearance(grid_);

    RCLCPP_INFO(
      get_logger(), "收到地图: %dx%d, 分辨率 %.3f m, 膨胀 %.2f m",
      grid_.width, grid_.height, grid_.resolution, inflation_radius_);
    if (have_path_) 
    {
      replan();
    }
  }

  // /path 回调：缓存 A* 路径并触发规划
  void onPath(const nav_msgs::msg::Path::SharedPtr msg)
  {
    last_path_ = msg;
    have_path_ = true;
    RCLCPP_INFO(get_logger(), "收到 A* 路径: %zu 个点", msg->poses.size());
    replan();
  }

  // 规划入口：做输入有效性检查，失败时发布空轨迹（清除下游显示）
  void replan()
  {
    if (!last_path_ || last_path_->poses.size() < 2) 
    {
      RCLCPP_WARN(get_logger(), "路径少于 2 个点，清除轨迹");
      publishEmpty();
      return;
    }
    if (!grid_.ready || !inflated_grid_.ready) 
    {
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 3000, "等待 /map，暂时无法检查障碍");
      return;
    }
    if (!plan(*last_path_)) 
    {
      publishEmpty();
    }
  }

  // 把 nav_msgs::Path 转成内部 Vec2 序列，过滤非有限值
  std::vector<Vec2> pathToPoints(const nav_msgs::msg::Path & path) const
  {
    std::vector<Vec2> pts;
    pts.reserve(path.poses.size());
    for (const auto & ps : path.poses) 
    {
      const double x = ps.pose.position.x;
      const double y = ps.pose.position.y;
      if (std::isfinite(x) && std::isfinite(y)) 
      {
        pts.push_back(Vec2{x, y});
      }
    }
    return pts;
  }

  // 沿某一方向探测可通行距离，最多 max_rect_half_ 米，用于构造矩形可行域
  // 使用膨胀地图，margin 已在膨胀层中体现
  double freeExtent(const Vec2 & c, int dir_x, int dir_y) const
  {
    double extent = 0.0;
    const GridMap & g = inflated_grid_.ready ? inflated_grid_ : grid_;
    for (double s = g.resolution; s <= max_rect_half_ + 1e-9; s += g.resolution) 
    {
      const double x = c.x + static_cast<double>(dir_x) * s;
      const double y = c.y + static_cast<double>(dir_y) * s;
      if (!isFreeWorld(g, x, y, 0.0, occupied_thresh_)) 
      {
        break;
      }
      extent = s;
    }
    return extent;
  }

  // 以连接点为中心，向四个方向探测，得到最大安全矩形 [lx,ux] x [ly,uy]
  Box makeFreeBox(const Vec2 & c) const
  {
    const double xm = freeExtent(c, -1, 0);
    const double xp = freeExtent(c, 1, 0);
    const double ym = freeExtent(c, 0, -1);
    const double yp = freeExtent(c, 0, 1);
    return Box{c.x - xm, c.x + xp, c.y - ym, c.y + yp};
  }

  // 矩形任一边小于 2*min_rect_half_ 时认为区域过窄，应把该连接点固定
  bool boxTooSmall(const Box & b) const
  {
    return (b.ux - b.lx) < 2.0 * min_rect_half_ ||
           (b.uy - b.ly) < 2.0 * min_rect_half_;
  }

  // 对整条分段轨迹按 collision_dt_ 密集采样做碰撞检测；
  // 返回 false 时通过 bad_segment 带回第一个撞障碍的段号。
  // 注意：这里查的是膨胀地图，margin 传 0，因为安全距离已经烘进膨胀层了。
  bool isTrajectoryFree(const std::vector<Segment> & segments, int & bad_segment) const
  {
    for (size_t s = 0; s < segments.size(); ++s) 
    {
      const int samples = std::max(
        2, static_cast<int>(std::ceil(segments[s].T / collision_dt_)));
      for (int i = 0; i <= samples; ++i) 
      {
        const double t = segments[s].T * static_cast<double>(i) / static_cast<double>(samples);
        const double tau = t / segments[s].T;
        const double x = evalPoly(segments[s].cx, 0, tau, segments[s].T);
        const double y = evalPoly(segments[s].cy, 0, tau, segments[s].T);
        // 用膨胀地图判断，等价于要求轨迹离真实障碍至少 inflation_radius_
        const GridMap & g = inflated_grid_.ready ? inflated_grid_ : grid_;
        if (!isFreeWorld(g, x, y, 0.0, occupied_thresh_)) 
        {
          bad_segment = static_cast<int>(s);
          return false;
        }
      }
    }
    return true;
  }

  // 核心规划函数：连接点提取 -> 时间分配 -> QP 求解 -> 碰撞检测与迭代收紧
  bool plan(const nav_msgs::msg::Path & path)
  {
    const auto t_start = std::chrono::steady_clock::now();

    // 步骤 1：路径预处理。先把 A* 路径限制在膨胀地图之外；
    // 如果原路径穿过膨胀区，就在膨胀地图上重新生成一条参考路径。
    const auto raw = pathToPoints(path);
    bool rerouted = false;
    const auto reference = auto_reroute_ ?
      makeReferencePath(
        raw, inflated_grid_, clearance_, occupied_thresh_,
        clearance_weight_, inflation_radius_ + 0.20, rerouted) :
      raw;
    if (rerouted) 
    {
      RCLCPP_INFO(
        get_logger(), "A* 路径穿过膨胀区，已在膨胀地图上重新生成参考路径 (%zu -> %zu 点)",
        raw.size(), reference.size());
    }

    // 连接点提取也改用膨胀地图做视线检查：
    // 这样简化后剩下的每一段折线都不会穿过膨胀层。
    const auto waypoints = extractWaypoints(
      reference, inflated_grid_, occupied_thresh_, insert_spacing_, 0.0);
    if (waypoints.size() < 2) 
    {
      RCLCPP_ERROR(get_logger(), "有效路径点不足，无法优化");
      return false;
    }

    // 步骤 2：时间分配。按段长 / 参考速度估算时长，并夹在 [min_seg_time_, max_seg_time_]
    const int N = static_cast<int>(waypoints.size()) - 1;   // 段数
    std::vector<double> T(static_cast<size_t>(N), min_seg_time_);
    for (int s = 0; s < N; ++s) 
    {
      const double d = norm(waypoints[static_cast<size_t>(s + 1)] - waypoints[static_cast<size_t>(s)]);
      T[static_cast<size_t>(s)] = std::clamp(
        d / v_ref_, min_seg_time_, max_seg_time_);
    }

    // 步骤 3：为每个连接点构造矩形可行域。
    // 起终点固定；内部点不再因为“轴对齐矩形过小”就被硬固定，
    // 否则窄通道处会退化成折线。膨胀层已经保证参考点在安全区域内。
    std::vector<Box> boxes(waypoints.size());
    std::vector<bool> fixed(waypoints.size(), false);
    std::vector<bool> stop(waypoints.size(), false);
    for (size_t k = 0; k < waypoints.size(); ++k) 
    {
      boxes[k] = makeFreeBox(waypoints[k]);
      if (k == 0 || k + 1 == waypoints.size()) 
      {
        fixed[k] = true;
      }
    }

    // 每段 tau = 0.25/0.5/0.75 的采样可行域，用于防止段内鼓出膨胀层。
    // 做法：在参考折线上取这三个插值点，各自向四周探测得到一个安全矩形，
    // 后面 buildBoxInequalities() 会把它们变成位置不等式。
    static const double sample_taus[3] = {0.25, 0.5, 0.75};
    std::vector<std::vector<Box>> sample_boxes(static_cast<size_t>(N));
    for (int s = 0; s < N; ++s) 
    {
      sample_boxes[static_cast<size_t>(s)].resize(3);
      const Vec2 & a = waypoints[static_cast<size_t>(s)];
      const Vec2 & b = waypoints[static_cast<size_t>(s + 1)];
      for (int j = 0; j < 3; ++j) 
      {
        const Vec2 p = a + (b - a) * sample_taus[j];
        sample_boxes[static_cast<size_t>(s)][static_cast<size_t>(j)] = makeFreeBox(p);
      }
    }

    RCLCPP_INFO(
      get_logger(), "优化输入: %zu 个 A* 点 -> %zu 个参考点 -> %zu 个连接点, %d 段, 总时长 %.2f s",
      raw.size(), reference.size(), waypoints.size(), N,
      std::accumulate(T.begin(), T.end(), 0.0));

    // 步骤 4：迭代求解。x、y 两轴共用 H 与等式约束矩阵，分别求解两个 QP。
    // H 依赖各段时长 T；碰撞时优先延长 T，因此每轮都要重建 H。
    for (int iter = 0; iter < max_iterations_; ++iter) 
    {
      const int nvar = 8 * N;
      Eigen::MatrixXd H = Eigen::MatrixXd::Zero(nvar, nvar);
      for (int s = 0; s < N; ++s) 
      {
        H.block<kCoeffs, kCoeffs>(8 * s, 8 * s) =
          snapUnitHessian() / std::pow(T[static_cast<size_t>(s)], 7.0);
      }
      // 加极小正则项，改善病态性（不影响最优解）
      const double hscale = H.diagonal().maxCoeff();
      if (hscale > 0.0) 
      {
        H.diagonal().array() += 1e-9 * hscale;
      }

      Eigen::MatrixXd Aeq;
      Eigen::VectorXd bx;
      Eigen::VectorXd by;
      buildEquality(T, waypoints, fixed, stop, Aeq, bx, by);   // 位置/连续性约束

      IneqSet ix;
      IneqSet iy;
      buildBoxInequalities(
        N, boxes, fixed, sample_boxes, ix, iy);                // 连接点 + 段内采样约束

      Eigen::VectorXd sol_x;
      Eigen::VectorXd sol_y;
      if (!solveBoxQP(H, Aeq, bx, ix, sol_x) ||
        !solveBoxQP(H, Aeq, by, iy, sol_y))
      {
        // 求解失败时退化为全部固定连接点，下一轮再试
        RCLCPP_ERROR(get_logger(), "QP 求解失败（第 %d 次迭代）", iter + 1);
        fixed.assign(waypoints.size(), true);
        continue;
      }

      // 把解写回分段多项式结构
      std::vector<Segment> segments(static_cast<size_t>(N));
      for (int s = 0; s < N; ++s) 
      {
        segments[static_cast<size_t>(s)].T = T[static_cast<size_t>(s)];
        segments[static_cast<size_t>(s)].cx = sol_x.segment<kCoeffs>(8 * s);
        segments[static_cast<size_t>(s)].cy = sol_y.segment<kCoeffs>(8 * s);
      }

      // 步骤 6：碰撞检测。通过则发布结果并结束；否则收紧约束重解。
      int bad = -1;
      if (isTrajectoryFree(segments, bad)) 
      {
        publishResult(segments);
        const double ms = std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - t_start).count();
        RCLCPP_INFO(
          get_logger(), "Minimum snap 成功: %d 段, %zu 连接点, 用时 %.1f ms",
          N, waypoints.size(), ms);
        return true;
      }

      // 平滑策略：碰撞时优先延长该段的时间，让曲线用更长时间绕过去，
      // 而不是把连接点钉死或加 v=a=j=0 的停止约束。
      // 时间变长后多项式有更“陡”的自由度，可以在不撞膨胀层的前提下保持连续。
      const double oldT = T[static_cast<size_t>(bad)];
      const double newT = std::min(max_seg_time_, oldT * 1.5 + 0.1);
      if (newT > oldT + 1e-6) 
      {
        T[static_cast<size_t>(bad)] = newT;
        // 相邻两段也稍微放长一点，避免局部时间突变导致速度/加速度跳变
        if (bad > 0) 
        {
          T[static_cast<size_t>(bad - 1)] = std::min(
            max_seg_time_, T[static_cast<size_t>(bad - 1)] * 1.1 + 0.05);
        }
        if (bad + 1 < N) 
        {
          T[static_cast<size_t>(bad + 1)] = std::min(
            max_seg_time_, T[static_cast<size_t>(bad + 1)] * 1.1 + 0.05);
        }
        RCLCPP_WARN(
          get_logger(), "第 %d 次迭代: 第 %d 段撞到膨胀层，延长该段时长 %.2f -> %.2f s",
          iter + 1, bad, oldT, newT);
        continue;
      }

      // 时间已到上限，才退化为收紧矩形；仍然不加停止约束。
      bool changed = false;
      const int knots[2] = {bad, bad + 1};
      for (int kk : knots) 
      {
        if (kk <= 0 || kk >= N) 
        {
          continue;
        }
        if (!fixed[static_cast<size_t>(kk)]) 
        {
          Box & b = boxes[static_cast<size_t>(kk)];
          const Vec2 c = waypoints[static_cast<size_t>(kk)];
          b.lx = c.x + (b.lx - c.x) * 0.5;
          b.ux = c.x + (b.ux - c.x) * 0.5;
          b.ly = c.y + (b.ly - c.y) * 0.5;
          b.uy = c.y + (b.uy - c.y) * 0.5;
          if (boxTooSmall(b)) 
          {
            fixed[static_cast<size_t>(kk)] = true;
            RCLCPP_INFO(
              get_logger(), "连接点 %d 已固定位置（不加停止约束）", kk);
          }
          changed = true;
        }
      }
      if (!changed) 
      {
        RCLCPP_ERROR(get_logger(), "时间与矩形都已无法继续调整，仍无法避开膨胀层");
        return false;
      }
    }

    RCLCPP_ERROR(get_logger(), "超过最大迭代次数，轨迹仍碰撞，放弃本次规划");
    return false;
  }

  // 按 sample_dt_ 对每段轨迹采样，发布平滑路径与含速度/加速度的轨迹消息
  void publishResult(const std::vector<Segment> & segments)
  {
    const auto stamp = now();

    nav_msgs::msg::Path path;
    path.header.stamp = stamp;
    path.header.frame_id = grid_.frame_id;

    trajectory_msgs::msg::MultiDOFJointTrajectory traj;
    traj.header.stamp = stamp;
    traj.header.frame_id = grid_.frame_id;
    traj.joint_names.push_back("base_link");

    double t0 = 0.0;        // 当前段之前的累计时间
    double last_yaw = 0.0;  // 速度过小时沿用上一时刻的朝向

    for (size_t s = 0; s < segments.size(); ++s) 
    {
      const Segment & seg = segments[s];
      const int samples = std::max(1, static_cast<int>(std::ceil(seg.T / sample_dt_)));
      for (int i = 0; i <= samples; ++i)
      {
        if (s > 0 && i == 0) 
        {
          continue;  // 衔接点已由上一段输出，避免重复
        }
        const double t = seg.T * static_cast<double>(i) / static_cast<double>(samples);
        const double tau = t / seg.T;

        // 位置、速度、加速度均由多项式求导得到
        const double px = evalPoly(seg.cx, 0, tau, seg.T);
        const double py = evalPoly(seg.cy, 0, tau, seg.T);
        const double vx = evalPoly(seg.cx, 1, tau, seg.T);
        const double vy = evalPoly(seg.cy, 1, tau, seg.T);
        const double ax = evalPoly(seg.cx, 2, tau, seg.T);
        const double ay = evalPoly(seg.cy, 2, tau, seg.T);

        // 由速度方向确定偏航角，再转成四元数
        if (std::hypot(vx, vy) > 1e-4) 
        {
          last_yaw = std::atan2(vy, vx);
        }
        const double half_yaw = 0.5 * last_yaw;

        geometry_msgs::msg::PoseStamped pose;
        pose.header = path.header;
        pose.pose.position.x = px;
        pose.pose.position.y = py;
        pose.pose.position.z = 0.0;
        pose.pose.orientation.x = 0.0;
        pose.pose.orientation.y = 0.0;
        pose.pose.orientation.z = std::sin(half_yaw);
        pose.pose.orientation.w = std::cos(half_yaw);
        path.poses.push_back(pose);

        trajectory_msgs::msg::MultiDOFJointTrajectoryPoint point;
        geometry_msgs::msg::Transform transform;
        transform.translation.x = px;
        transform.translation.y = py;
        transform.translation.z = 0.0;
        transform.rotation = pose.pose.orientation;
        point.transforms.push_back(transform);

        geometry_msgs::msg::Twist velocity;
        velocity.linear.x = vx;
        velocity.linear.y = vy;
        point.velocities.push_back(velocity);

        geometry_msgs::msg::Twist acceleration;
        acceleration.linear.x = ax;
        acceleration.linear.y = ay;
        point.accelerations.push_back(acceleration);

        point.time_from_start = rclcpp::Duration::from_seconds(t0 + t);
        traj.points.push_back(point);
      }
      t0 += seg.T;
    }

    RCLCPP_INFO(
      get_logger(), "发布轨迹: %zu 个采样点, 总时长 %.2f s",
      path.poses.size(), t0);

    pub_path_->publish(path);
    pub_traj_->publish(traj);
  }

  // 发布空消息，用于规划失败或输入无效时清除下游显示
  void publishEmpty()
  {
    const auto stamp = now();
    nav_msgs::msg::Path path;
    path.header.stamp = stamp;
    path.header.frame_id = grid_.ready ? grid_.frame_id : "map";
    pub_path_->publish(path);

    trajectory_msgs::msg::MultiDOFJointTrajectory traj;
    traj.header.stamp = stamp;
    traj.header.frame_id = path.header.frame_id;
    pub_traj_->publish(traj);
  }

  // ---------------- 参数 ----------------
  double v_ref_ = 1.0;              // 参考速度 (m/s)
  double min_seg_time_ = 0.5;       // 单段最短时长 (s)
  double max_seg_time_ = 8.0;       // 单段最长时长 (s)
  double insert_spacing_ = 2.0;     // 长直线插点间距 (m)
  double max_rect_half_ = 0.30;     // 矩形最大半边长 (m)
  double min_rect_half_ = 0.03;     // 矩形最小半边长 (m)
  double obstacle_margin_ = 0.03;   // 碰撞检测裕度 (m)
  double sample_dt_ = 0.02;         // 轨迹发布采样周期 (s)
  double collision_dt_ = 0.01;      // 碰撞检测采样周期 (s)
  int max_iterations_ = 20;         // 最大收紧迭代次数
  int occupied_thresh_ = 50;        // 占据阈值
  double inflation_radius_ = 0.35;  // 轨迹优化膨胀半径 (m)
  double clearance_weight_ = 0.5;   // 参考路径居中权重
  bool auto_reroute_ = true;        // 穿过膨胀区时自动重找参考路径

  // ---------------- 运行状态 ----------------
  GridMap grid_;                    // 原始地图缓存
  GridMap inflated_grid_;           // 轨迹优化专用膨胀地图
  std::vector<float> clearance_;    // 每格到最近障碍的距离 (m)
  nav_msgs::msg::Path::SharedPtr last_path_;  // 最近一次 A* 路径
  bool have_path_ = false;          // 是否已收到路径

  // ---------------- ROS 接口 ----------------
  rclcpp::Subscription<nav_msgs::msg::OccupancyGrid>::SharedPtr map_sub_;
  rclcpp::Subscription<nav_msgs::msg::Path>::SharedPtr path_sub_;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr pub_path_;
  rclcpp::Publisher<trajectory_msgs::msg::MultiDOFJointTrajectory>::SharedPtr pub_traj_;
};

// 程序入口：初始化 ROS 并进入回调循环（收到 /map 与 /path 后自动规划）
int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<MinSnapPlanner>());
  rclcpp::shutdown();
  return 0;
}
