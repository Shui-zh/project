// 输入： /map  (nav_msgs/OccupancyGrid)
//       /path (nav_msgs/Path，A* 输出的折线，只取起点和终点作为约束)
// 输出： /min_snap_path (nav_msgs/Path)
//       /trajectory    (trajectory_msgs/MultiDOFJointTrajectory)


#include "rclcpp/rclcpp.hpp"
#include "nav_msgs/msg/path.hpp"
#include "nav_msgs/msg/occupancy_grid.hpp"
#include "trajectory_msgs/msg/multi_dof_joint_trajectory.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "geometry_msgs/msg/transform.hpp"
#include "geometry_msgs/msg/twist.hpp"

#include <Eigen/Dense>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <queue>
#include <string>
#include <utility>
#include <vector>

namespace
{

// 7 次多项式：每段 8 个系数，可保证 snap（4 阶导）分段连续
constexpr int kOrder = 7;
constexpr int kCoeffs = kOrder + 1;
using Vec8 = Eigen::Matrix<double, kCoeffs, 1>;
using Row8 = Eigen::Matrix<double, 1, kCoeffs>;

struct Vec2
{
  double x = 0.0;
  double y = 0.0;
};

Vec2 operator+(const Vec2 & a, const Vec2 & b) {return Vec2{a.x + b.x, a.y + b.y};}
Vec2 operator-(const Vec2 & a, const Vec2 & b) {return Vec2{a.x - b.x, a.y - b.y};}
Vec2 operator*(const Vec2 & a, double s) {return Vec2{a.x * s, a.y * s};}
double norm(const Vec2 & a) {return std::hypot(a.x, a.y);}

// 栅格地图（从 /map 拷贝出来的轻量结构）
struct GridMap
{
  std::vector<int8_t> data;
  int width = 0;
  int height = 0;
  double resolution = 0.05;
  double origin_x = 0.0;
  double origin_y = 0.0;
  std::string frame_id = "map";
  bool ready = false;
};

// 一段轨迹：时长 T + x/y 两个方向的归一化时间多项式系数
struct Segment
{
  double T = 1.0;
  Vec8 cx = Vec8::Zero();
  Vec8 cy = Vec8::Zero();
};

// ----------------------------------------------------------------------------
// 距离场（clearance）：每个栅格到最近障碍的欧氏距离，单位米
//
// 用精确的欧氏距离变换（两遍一维距离变换），比 8 邻域 Dijkstra 更圆、
// 更适合用来判断“离障碍有多远”。
// ----------------------------------------------------------------------------

void distanceTransform1D(const std::vector<double> & f, std::vector<double> & d)
{
  const int n = static_cast<int>(f.size());
  d.assign(static_cast<size_t>(n), 0.0);
  if (n == 0) {
    return;
  }

  std::vector<int> v(static_cast<size_t>(n), 0);
  std::vector<double> z(static_cast<size_t>(n) + 1, 0.0);

  int k = 0;
  v[0] = 0;
  z[0] = -1e30;
  z[1] = 1e30;

  for (int q = 1; q < n; ++q) {
    double s = ((f[q] + static_cast<double>(q) * q) -
      (f[v[k]] + static_cast<double>(v[k]) * v[k])) /
      (2.0 * q - 2.0 * v[k]);
    while (s <= z[k]) {
      --k;
      s = ((f[q] + static_cast<double>(q) * q) -
        (f[v[k]] + static_cast<double>(v[k]) * v[k])) /
        (2.0 * q - 2.0 * v[k]);
    }
    ++k;
    v[k] = q;
    z[k] = s;
    z[k + 1] = 1e30;
  }

  k = 0;
  for (int q = 0; q < n; ++q) {
    while (z[k + 1] < q) {
      ++k;
    }
    const double diff = static_cast<double>(q - v[k]);
    d[q] = diff * diff + f[v[k]];
  }
}

std::vector<float> buildClearance(const GridMap & g, int occupied_thresh)
{
  const int w = g.width;
  const int h = g.height;
  std::vector<float> out(static_cast<size_t>(w) * h, 0.0f);
  if (w <= 0 || h <= 0 || g.resolution <= 0.0) {
    return out;
  }

  const double inf = 1e12;
  std::vector<double> grid(static_cast<size_t>(w) * h, inf);
  for (size_t i = 0; i < grid.size(); ++i) {
    if (g.data[i] >= occupied_thresh) {
      grid[i] = 0.0;
    }
  }

  // 先按列做一维距离变换，再按行做，得到精确的二维欧氏距离平方
  std::vector<double> f;
  std::vector<double> d;
  for (int x = 0; x < w; ++x) {
    f.resize(static_cast<size_t>(h));
    for (int y = 0; y < h; ++y) {
      f[static_cast<size_t>(y)] = grid[static_cast<size_t>(y) * w + x];
    }
    distanceTransform1D(f, d);
    for (int y = 0; y < h; ++y) {
      grid[static_cast<size_t>(y) * w + x] = d[static_cast<size_t>(y)];
    }
  }
  for (int y = 0; y < h; ++y) {
    f.resize(static_cast<size_t>(w));
    for (int x = 0; x < w; ++x) {
      f[static_cast<size_t>(x)] = grid[static_cast<size_t>(y) * w + x];
    }
    distanceTransform1D(f, d);
    for (int x = 0; x < w; ++x) {
      grid[static_cast<size_t>(y) * w + x] = d[static_cast<size_t>(x)];
    }
  }

  const double max_dist = std::hypot(static_cast<double>(w), static_cast<double>(h)) *
    g.resolution;
  for (size_t i = 0; i < grid.size(); ++i) {
    double dist = std::sqrt(std::max(0.0, grid[i])) * g.resolution;
    if (!std::isfinite(dist)) {
      dist = max_dist;
    }
    out[i] = static_cast<float>(std::min(dist, max_dist));
  }
  return out;
}

// 世界坐标处的 clearance（米）
double clearanceAt(
  const GridMap & g, const std::vector<float> & clearance, double x, double y)
{
  if (g.width <= 0 || g.height <= 0 || g.resolution <= 0.0) {
    return 0.0;
  }
  const int gx = static_cast<int>(std::floor((x - g.origin_x) / g.resolution));
  const int gy = static_cast<int>(std::floor((y - g.origin_y) / g.resolution));
  if (gx < 0 || gy < 0 || gx >= g.width || gy >= g.height) {
    return 0.0;
  }
  return static_cast<double>(clearance[static_cast<size_t>(gy) * g.width + gx]);
}

// 线段 a->b 上所有采样点的 clearance 是否都 >= required
bool segmentClearanceOK(
  const GridMap & g, const std::vector<float> & clearance,
  const Vec2 & a, const Vec2 & b, double required)
{
  const double len = norm(b - a);
  const int samples = std::max(
    2, static_cast<int>(std::ceil(len / (g.resolution * 0.5))));
  for (int i = 0; i <= samples; ++i) {
    const double t = static_cast<double>(i) / static_cast<double>(samples);
    const Vec2 p = a + (b - a) * t;
    if (clearanceAt(g, clearance, p.x, p.y) < required) {
      return false;
    }
  }
  return true;
}

// 从 (cx, cy) 出发，用 BFS 找最近的、clearance >= required 的栅格
bool nearestCellWithClearance(
  const GridMap & g, const std::vector<float> & clearance,
  int & cx, int & cy, double required)
{
  if (g.width <= 0 || g.height <= 0) {
    return false;
  }
  const int w = g.width;
  const int h = g.height;
  cx = std::clamp(cx, 0, w - 1);
  cy = std::clamp(cy, 0, h - 1);

  const auto ok = [&](int x, int y) {
      return clearance[static_cast<size_t>(y) * w + x] >= required;
    };
  if (ok(cx, cy)) {
    return true;
  }

  std::vector<char> visited(static_cast<size_t>(w) * h, 0);
  std::queue<std::pair<int, int>> q;
  q.emplace(cx, cy);
  visited[static_cast<size_t>(cy) * w + cx] = 1;

  static const int dx[8] = {1, -1, 0, 0, 1, 1, -1, -1};
  static const int dy[8] = {0, 0, 1, -1, 1, -1, 1, -1};

  while (!q.empty()) {
    const auto cur = q.front();
    q.pop();
    for (int k = 0; k < 8; ++k) {
      const int nx = cur.first + dx[k];
      const int ny = cur.second + dy[k];
      if (nx < 0 || ny < 0 || nx >= w || ny >= h) {
        continue;
      }
      const size_t ni = static_cast<size_t>(ny) * w + nx;
      if (visited[ni]) {
        continue;
      }
      if (ok(nx, ny)) {
        cx = nx;
        cy = ny;
        return true;
      }
      visited[ni] = 1;
      q.emplace(nx, ny);
    }
  }
  return false;
}

// ----------------------------------------------------------------------------
// 参考路径：带 clearance 代价的 A*
//
// 与 astar_planner_demo 的区别只有代价函数：
//   真实步长 + 权重 * (目标间隙 - 当前间隙)^2
// 这样路径会主动离开障碍，尽量走通道中间，从而绕开夹缝，而不是贴边穿过去。
// ----------------------------------------------------------------------------

std::vector<Vec2> searchGuide(
  const GridMap & g, const std::vector<float> & clearance,
  const Vec2 & start, const Vec2 & goal,
  double required_clearance, double target_clearance, double clearance_weight)
{
  std::vector<Vec2> empty;
  if (g.width <= 0 || g.height <= 0 || g.resolution <= 0.0) {
    return empty;
  }

  const auto toCell = [&](const Vec2 & p, int & x, int & y) {
      x = static_cast<int>(std::floor((p.x - g.origin_x) / g.resolution));
      y = static_cast<int>(std::floor((p.y - g.origin_y) / g.resolution));
    };

  int sx = 0;
  int sy = 0;
  int gx = 0;
  int gy = 0;
  toCell(start, sx, sy);
  toCell(goal, gx, gy);
  if (!nearestCellWithClearance(g, clearance, sx, sy, required_clearance) ||
    !nearestCellWithClearance(g, clearance, gx, gy, required_clearance))
  {
    return empty;
  }

  const int w = g.width;
  const int h = g.height;
  const int n = w * h;
  const int si = sy * w + sx;
  const int gi = gy * w + gx;
  const double inf = std::numeric_limits<double>::infinity();
  const double target = std::max(0.05, target_clearance);

  // 八邻域 octile 启发，乘分辨率换算成米
  const auto heuristic = [&](int x, int y) {
      const int adx = std::abs(x - gx);
      const int ady = std::abs(y - gy);
      const int mn = std::min(adx, ady);
      const int mx = std::max(adx, ady);
      return (mx + (std::sqrt(2.0) - 1.0) * mn) * g.resolution;
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

  while (!open.empty()) {
    const int cur = open.top().second;
    open.pop();
    if (closed[static_cast<size_t>(cur)]) {
      continue;
    }
    closed[static_cast<size_t>(cur)] = 1;
    if (cur == gi) {
      break;
    }

    const int cx = cur % w;
    const int cy = cur / w;
    for (int k = 0; k < 8; ++k) {
      const int nx = cx + dx[k];
      const int ny = cy + dy[k];
      if (nx < 0 || ny < 0 || nx >= w || ny >= h) {
        continue;
      }

      const double d = static_cast<double>(clearance[static_cast<size_t>(ny) * w + nx]);
      if (d < required_clearance) {
        continue;
      }
      // 斜向移动禁止切障碍角
      if (dx[k] != 0 && dy[k] != 0 &&
        (clearance[static_cast<size_t>(cy) * w + (cx + dx[k])] < required_clearance ||
        clearance[static_cast<size_t>(cy + dy[k]) * w + cx] < required_clearance))
      {
        continue;
      }

      const int ni = ny * w + nx;
      if (closed[static_cast<size_t>(ni)]) {
        continue;
      }

      const double step = (dx[k] != 0 && dy[k] != 0) ?
        g.resolution * std::sqrt(2.0) : g.resolution;
      // 间隙惩罚：离障碍越近，代价越大；超过 target 后不再奖励
      const double gap = std::max(0.0, target - d) / target;
      const double penalty = clearance_weight * gap * gap;
      const double ng = gscore[static_cast<size_t>(cur)] + step * (1.0 + penalty);

      if (ng < gscore[static_cast<size_t>(ni)]) {
        gscore[static_cast<size_t>(ni)] = ng;
        parent[static_cast<size_t>(ni)] = cur;
        open.emplace(ng + heuristic(nx, ny), ni);
      }
    }
  }

  if (gi != si && parent[static_cast<size_t>(gi)] < 0) {
    return empty;
  }

  std::vector<Vec2> path;
  for (int cur = gi; cur >= 0; cur = parent[static_cast<size_t>(cur)]) {
    const int x = cur % w;
    const int y = cur / w;
    path.push_back(Vec2{
      g.origin_x + (static_cast<double>(x) + 0.5) * g.resolution,
      g.origin_y + (static_cast<double>(y) + 0.5) * g.resolution});
    if (cur == si) {
      break;
    }
  }
  std::reverse(path.begin(), path.end());

  // 保证轨迹严格从原始起点出发、到原始终点结束
  if (norm(path.front() - start) > 1e-6) {
    path.insert(path.begin(), start);
  }
  if (norm(path.back() - goal) > 1e-6) {
    path.push_back(goal);
  }
  return path;
}

// 去掉相邻重复点
void dedupe(std::vector<Vec2> & pts)
{
  std::vector<Vec2> out;
  out.reserve(pts.size());
  for (const auto & p : pts) {
    if (out.empty() || norm(p - out.back()) > 1e-6) {
      out.push_back(p);
    }
  }
  pts.swap(out);
}

// 贪心视线简化：从当前点直接连到最远的、全程 clearance 足够的点
std::vector<Vec2> simplifyGuide(
  const std::vector<Vec2> & path, const GridMap & g,
  const std::vector<float> & clearance, double required)
{
  std::vector<Vec2> out;
  if (path.size() < 2) {
    return path;
  }
  out.push_back(path.front());

  size_t i = 0;
  while (i + 1 < path.size()) {
    size_t j = path.size() - 1;
    while (j > i + 1 && !segmentClearanceOK(g, clearance, path[i], path[j], required)) {
      --j;
    }
    out.push_back(path[j]);
    i = j;
  }
  return out;
}

// 按间距重采样，给 minimum snap 提供足够多的连接点
std::vector<Vec2> resampleGuide(const std::vector<Vec2> & in, double spacing)
{
  std::vector<Vec2> out;
  if (in.size() < 2) {
    return in;
  }
  const double step = std::max(0.2, spacing);
  out.push_back(in.front());

  for (size_t k = 0; k + 1 < in.size(); ++k) {
    const Vec2 a = in[k];
    const Vec2 b = in[k + 1];
    const double len = norm(b - a);
    const int pieces = std::max(1, static_cast<int>(std::ceil(len / step)));
    for (int j = 1; j <= pieces; ++j) {
      const double t = static_cast<double>(j) / static_cast<double>(pieces);
      out.push_back(a + (b - a) * t);
    }
  }
  out.back() = in.back();
  dedupe(out);
  return out;
}

// ----------------------------------------------------------------------------
// Minimum Snap 数学部分
// ----------------------------------------------------------------------------

// 归一化时间 tau 下的 d 阶导数行向量
Row8 derivRow(int d, double tau)
{
  Row8 r = Row8::Zero();
  for (int i = d; i < kCoeffs; ++i) {
    double factor = 1.0;
    for (int k = 0; k < d; ++k) {
      factor *= static_cast<double>(i - k);
    }
    r[i] = factor * std::pow(tau, i - d);
  }
  return r;
}

// 归一化时间下 snap 平方积分的 Hessian
const Eigen::Matrix<double, kCoeffs, kCoeffs> & snapUnitHessian()
{
  static const Eigen::Matrix<double, kCoeffs, kCoeffs> q = [] {
      Eigen::Matrix<double, kCoeffs, kCoeffs> m =
        Eigen::Matrix<double, kCoeffs, kCoeffs>::Zero();
      for (int i = 4; i < kCoeffs; ++i) {
        for (int j = 4; j < kCoeffs; ++j) {
          double ci = 1.0;
          double cj = 1.0;
          for (int k = 0; k < 4; ++k) {
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

// 计算某段多项式在 tau 处的 d 阶导数（换算回真实时间）
double evalPoly(const Vec8 & c, int derivative, double tau, double T)
{
  return derivRow(derivative, tau).dot(c) / std::pow(T, derivative);
}

// 组装等式约束：
//   1) 起点位置固定，速度/加速度/jerk = 0；
//   2) 终点位置固定，速度/加速度/jerk = 0；
//   3) 相邻段在连接点处 0~4 阶导数连续（位置、速度、加速度、jerk、snap）；
//   4) 每个内部连接点位置固定到参考路径点上。
//
// 因为没有“窄处强行固定 + 停止约束”，曲线不会在夹缝里退化成折线。
void buildEquality(
  const std::vector<double> & T, const std::vector<Vec2> & wp,
  Eigen::MatrixXd & A, Eigen::VectorXd & bx, Eigen::VectorXd & by)
{
  const int N = static_cast<int>(T.size());
  const int n = kCoeffs * N;

  std::vector<Eigen::RowVectorXd> rows;
  rows.reserve(static_cast<size_t>(8 + 5 * (N - 1) + (N - 1)));
  std::vector<double> rhs_x;
  std::vector<double> rhs_y;

  const auto push = [&](const Eigen::RowVectorXd & row, double x, double y) {
      rows.push_back(row);
      rhs_x.push_back(x);
      rhs_y.push_back(y);
    };

  // 起点：位置 + 前三阶导数为 0
  for (int d = 0; d <= 3; ++d) {
    Eigen::RowVectorXd row = Eigen::RowVectorXd::Zero(n);
    row.segment<kCoeffs>(0) = derivRow(d, 0.0) / std::pow(T.front(), d);
    push(row, d == 0 ? wp.front().x : 0.0, d == 0 ? wp.front().y : 0.0);
  }

  // 终点：位置 + 前三阶导数为 0
  const int last = N - 1;
  for (int d = 0; d <= 3; ++d) {
    Eigen::RowVectorXd row = Eigen::RowVectorXd::Zero(n);
    row.segment<kCoeffs>(kCoeffs * last) =
      derivRow(d, 1.0) / std::pow(T.back(), d);
    push(row, d == 0 ? wp.back().x : 0.0, d == 0 ? wp.back().y : 0.0);
  }

  // 连接点连续性：d = 0..4，包含 snap，曲线更顺
  for (int k = 1; k < N; ++k) {
    for (int d = 0; d <= 4; ++d) {
      Eigen::RowVectorXd row = Eigen::RowVectorXd::Zero(n);
      row.segment<kCoeffs>(kCoeffs * (k - 1)) =
        derivRow(d, 1.0) / std::pow(T[static_cast<size_t>(k - 1)], d);
      row.segment<kCoeffs>(kCoeffs * k) =
        -derivRow(d, 0.0) / std::pow(T[static_cast<size_t>(k)], d);
      push(row, 0.0, 0.0);
    }
  }

  // 内部连接点位置固定（参考点本身已经离障碍足够远）
  for (int k = 1; k < N; ++k) {
    Eigen::RowVectorXd row = Eigen::RowVectorXd::Zero(n);
    row[kCoeffs * k] = 1.0;
    push(row, wp[static_cast<size_t>(k)].x, wp[static_cast<size_t>(k)].y);
  }

  const int m = static_cast<int>(rows.size());
  A.resize(m, n);
  bx.resize(m);
  by.resize(m);
  for (int r = 0; r < m; ++r) {
    A.row(r) = rows[static_cast<size_t>(r)];
    bx[r] = rhs_x[static_cast<size_t>(r)];
    by[r] = rhs_y[static_cast<size_t>(r)];
  }
}

// 等式约束 QP：min 0.5 x^T H x  s.t. A x = b
// 通过 KKT 矩阵 [H A^T; A 0] 的 QR 分解求解
bool solveEqualityQP(
  const Eigen::MatrixXd & H, const Eigen::MatrixXd & A, const Eigen::VectorXd & b,
  Eigen::VectorXd & x)
{
  const int n = static_cast<int>(H.rows());
  const int m = static_cast<int>(A.rows());

  Eigen::MatrixXd K = Eigen::MatrixXd::Zero(n + m, n + m);
  K.topLeftCorner(n, n) = H;

  Eigen::VectorXd rhs = Eigen::VectorXd::Zero(n + m);
  if (m > 0) {
    K.topRightCorner(n, m) = A.transpose();
    K.bottomLeftCorner(m, n) = A;
    rhs.tail(m) = b;
  }

  const Eigen::VectorXd sol = K.colPivHouseholderQr().solve(rhs);
  const double residual = (K * sol - rhs).norm();
  if (!std::isfinite(residual) || residual > 1e-5 * (1.0 + rhs.norm())) {
    return false;
  }
  x = sol.head(n);
  return true;
}

// 按 v_ref 分配每段时间，并夹在 [min_seg_time, max_seg_time]
void allocateTime(
  const std::vector<Vec2> & wp, double v_ref,
  double min_seg_time, double max_seg_time, std::vector<double> & T)
{
  const int N = static_cast<int>(wp.size()) - 1;
  T.assign(static_cast<size_t>(N), min_seg_time);
  for (int s = 0; s < N; ++s) {
    const double d = norm(wp[static_cast<size_t>(s + 1)] - wp[static_cast<size_t>(s)]);
    T[static_cast<size_t>(s)] = std::clamp(d / std::max(0.1, v_ref), min_seg_time, max_seg_time);
  }
}

// ----------------------------------------------------------------------------
// 碰撞检测与迭代调整
// ----------------------------------------------------------------------------

// 对整条轨迹按 collision_dt 密集采样，检查离障碍距离是否都 >= required
bool trajectoryClear(
  const std::vector<Segment> & segments, const GridMap & g,
  const std::vector<float> & clearance, double required, double collision_dt,
  int & bad_segment)
{
  for (size_t s = 0; s < segments.size(); ++s) {
    const int samples = std::max(
      2, static_cast<int>(std::ceil(segments[s].T / std::max(1e-3, collision_dt))));
    for (int i = 0; i <= samples; ++i) {
      const double tau = static_cast<double>(i) / static_cast<double>(samples);
      const double x = evalPoly(segments[s].cx, 0, tau, segments[s].T);
      const double y = evalPoly(segments[s].cy, 0, tau, segments[s].T);
      if (clearanceAt(g, clearance, x, y) < required) {
        bad_segment = static_cast<int>(s);
        return false;
      }
    }
  }
  return true;
}

// 时间已经到上限时，沿着 clearance 梯度把碰撞段两端的连接点往外推一点
bool nudgeWaypoints(
  std::vector<Vec2> & wp, const GridMap & g, const std::vector<float> & clearance,
  int bad, double required)
{
  bool moved = false;
  for (int k = bad; k <= bad + 1; ++k) {
    if (k <= 0 || k >= static_cast<int>(wp.size()) - 1) {
      continue;
    }
    const double h = std::max(1e-3, g.resolution * 0.5);
    const Vec2 p = wp[static_cast<size_t>(k)];
    const double dx = clearanceAt(g, clearance, p.x + h, p.y) -
      clearanceAt(g, clearance, p.x - h, p.y);
    const double dy = clearanceAt(g, clearance, p.x, p.y + h) -
      clearanceAt(g, clearance, p.x, p.y - h);
    const double len = std::hypot(dx, dy);
    if (len < 1e-9) {
      continue;
    }

    const double step = 0.08;
    const Vec2 cand{p.x + step * dx / len, p.y + step * dy / len};
    if (clearanceAt(g, clearance, cand.x, cand.y) >
      clearanceAt(g, clearance, p.x, p.y) + 1e-4)
    {
      wp[static_cast<size_t>(k)] = cand;
      moved = true;
    }
  }
  (void)required;
  return moved;
}

// ----------------------------------------------------------------------------
// ROS 节点
// ----------------------------------------------------------------------------

class MinSnapPlanner : public rclcpp::Node
{
public:
  MinSnapPlanner() : Node("min_snap")
  {
    // ---------------- 参数 ----------------
    occupied_thresh_ = declare_parameter<int>("occupied_thresh", 50);

    // 参考路径的离障要求：
    //   hard_clearance   ：路径必须满足的最小离障距离（硬约束）
    //   clearance_target ：希望达到的离障距离，越接近越靠近通道中间
    //   clearance_weight ：间隙代价权重，越大越愿意绕远路去走宽通道
    hard_clearance_ = declare_parameter<double>("hard_clearance", 0.35);
    clearance_target_ = declare_parameter<double>("clearance_target", 1.5);
    clearance_weight_ = declare_parameter<double>("clearance_weight", 4.0);
    guide_margin_ = declare_parameter<double>("guide_margin", 0.40);
    waypoint_spacing_ = declare_parameter<double>("waypoint_spacing", 1.2);

    v_ref_ = declare_parameter<double>("v_ref", 1.0);
    min_seg_time_ = declare_parameter<double>("min_seg_time", 1.0);
    max_seg_time_ = declare_parameter<double>("max_seg_time", 4.0);
    sample_dt_ = declare_parameter<double>("sample_dt", 0.02);
    collision_dt_ = declare_parameter<double>("collision_dt", 0.01);
    max_iters_ = declare_parameter<int>("max_iters", 30);

    // 参数保护
    hard_clearance_ = std::max(0.02, hard_clearance_);
    clearance_target_ = std::max(hard_clearance_, clearance_target_);
    clearance_weight_ = std::max(0.0, clearance_weight_);
    guide_margin_ = std::max(0.0, guide_margin_);
    waypoint_spacing_ = std::max(0.2, waypoint_spacing_);
    v_ref_ = std::max(0.1, v_ref_);
    min_seg_time_ = std::max(0.2, min_seg_time_);
    max_seg_time_ = std::max(min_seg_time_, max_seg_time_);
    sample_dt_ = std::max(0.005, sample_dt_);
    collision_dt_ = std::max(0.002, collision_dt_);
    max_iters_ = std::max(1, max_iters_);

    // ---------------- 通信 ----------------
    const auto map_qos = rclcpp::QoS(1).transient_local();
    const auto path_qos = rclcpp::QoS(1).transient_local();

    map_sub_ = create_subscription<nav_msgs::msg::OccupancyGrid>(
      "/map", map_qos,
      [this](const nav_msgs::msg::OccupancyGrid::SharedPtr msg) {onMap(msg);});

    path_sub_ = create_subscription<nav_msgs::msg::Path>(
      "/path", path_qos,
      [this](const nav_msgs::msg::Path::SharedPtr msg) {onPath(msg);});

    pub_path_ = create_publisher<nav_msgs::msg::Path>("/min_snap_path", path_qos);
    pub_traj_ = create_publisher<trajectory_msgs::msg::MultiDOFJointTrajectory>(
      "/trajectory", path_qos);

    RCLCPP_INFO(
      get_logger(),
      "Minimum snap 就绪: 硬离障 %.2f m, 目标离障 %.2f m, 权重 %.1f, 连接点间距 %.2f m",
      hard_clearance_, clearance_target_, clearance_weight_, waypoint_spacing_);
  }

private:
  // --------------------------------------------------------------------------
  // 输入回调
  // --------------------------------------------------------------------------
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

    // 距离场只在这里算一次，后续 A* 和碰撞检测都用它
    clearance_ = buildClearance(grid_, occupied_thresh_);

    RCLCPP_INFO(
      get_logger(), "收到地图: %dx%d, 分辨率 %.3f m",
      grid_.width, grid_.height, grid_.resolution);

    if (have_path_) {
      replan();
    }
  }

  void onPath(const nav_msgs::msg::Path::SharedPtr msg)
  {
    last_path_ = msg;
    have_path_ = true;
    RCLCPP_INFO(get_logger(), "收到 A* 路径: %zu 个点", msg->poses.size());
    replan();
  }

  // 把 A* 路径消息转成平面点
  static std::vector<Vec2> pathToPoints(const nav_msgs::msg::Path & path)
  {
    std::vector<Vec2> pts;
    pts.reserve(path.poses.size());
    for (const auto & ps : path.poses) {
      const double x = ps.pose.position.x;
      const double y = ps.pose.position.y;
      if (std::isfinite(x) && std::isfinite(y)) {
        pts.push_back(Vec2{x, y});
      }
    }
    return pts;
  }

  // --------------------------------------------------------------------------
  // 主流程
  // --------------------------------------------------------------------------
  void replan()
  {
    if (!last_path_ || last_path_->poses.size() < 2) {
      RCLCPP_WARN(get_logger(), "路径少于 2 个点，清除轨迹");
      publishEmpty();
      return;
    }
    if (!grid_.ready || clearance_.empty()) {
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 3000, "等待 /map，暂时无法检查障碍");
      return;
    }

    const auto raw = pathToPoints(*last_path_);
    if (raw.size() < 2) {
      RCLCPP_ERROR(get_logger(), "A* 路径无有效点");
      publishEmpty();
      return;
    }

    // 步骤 1：生成远离障碍的参考路径。
    // 硬离障要求从大到小依次尝试，保证在很窄的地图里也能找到可行路径。
    const Vec2 start = raw.front();
    const Vec2 goal = raw.back();
    std::vector<Vec2> guide;
    double used_clearance = hard_clearance_;
    const double relax[4] = {1.0, 0.7, 0.4, 0.15};
    for (double r : relax) {
      const double req = std::max(0.03, hard_clearance_ * r);
      guide = searchGuide(
        grid_, clearance_, start, goal, req, clearance_target_, clearance_weight_);
      if (guide.size() >= 2) {
        used_clearance = req;
        break;
      }
    }
    if (guide.size() < 2) {
      RCLCPP_ERROR(get_logger(), "无法生成远离障碍的参考路径");
      publishEmpty();
      return;
    }

    // 步骤 2：视线简化 + 等间距重采样。
    // 简化会保留“能直线通过且离障足够远”的长段，因此曲线不会被夹缝带偏。
    const double los_required = used_clearance + guide_margin_;
    auto simplified = simplifyGuide(guide, grid_, clearance_, los_required);
    auto waypoints = resampleGuide(simplified, waypoint_spacing_);
    dedupe(waypoints);
    if (waypoints.size() < 2) {
      RCLCPP_ERROR(get_logger(), "有效连接点不足");
      publishEmpty();
      return;
    }

    // 步骤 3：minimum snap。
    // 检查离障从 used_clearance 开始，失败再逐级放宽，避免端点本身离墙太近时永远失败。
    const double checks[4] = {
      used_clearance,
      used_clearance * 0.75,
      used_clearance * 0.5,
      0.05};
    for (double check : checks) {
      if (plan(waypoints, std::max(0.03, check))) {
        return;
      }
    }

    RCLCPP_ERROR(get_logger(), "Minimum snap 失败，清除轨迹");
    publishEmpty();
  }

  // --------------------------------------------------------------------------
  // minimum snap 求解 + 碰撞迭代
  // --------------------------------------------------------------------------
  // waypoints 按值传入：碰撞时可能会微调连接点，不影响调用方的数据
  bool plan(std::vector<Vec2> waypoints, double check_clearance)
  {
    const auto t_start = std::chrono::steady_clock::now();
    const int N = static_cast<int>(waypoints.size()) - 1;
    if (N < 1) {
      return false;
    }

    std::vector<double> T;
    allocateTime(waypoints, v_ref_, min_seg_time_, max_seg_time_, T);

    for (int iter = 0; iter < max_iters_; ++iter) {
      // H 依赖每段时长，时间一变就要重建
      const int nvar = kCoeffs * N;
      Eigen::MatrixXd H = Eigen::MatrixXd::Zero(nvar, nvar);
      for (int s = 0; s < N; ++s) {
        H.block<kCoeffs, kCoeffs>(kCoeffs * s, kCoeffs * s) =
          snapUnitHessian() / std::pow(T[static_cast<size_t>(s)], 7.0);
      }
      const double hscale = H.diagonal().maxCoeff();
      if (hscale > 0.0) {
        H.diagonal().array() += 1e-9 * hscale;
      }

      Eigen::MatrixXd A;
      Eigen::VectorXd bx;
      Eigen::VectorXd by;
      buildEquality(T, waypoints, A, bx, by);

      Eigen::VectorXd sol_x;
      Eigen::VectorXd sol_y;
      if (!solveEqualityQP(H, A, bx, sol_x) ||
        !solveEqualityQP(H, A, by, sol_y))
      {
        RCLCPP_ERROR(get_logger(), "QP 求解失败（第 %d 次迭代）", iter + 1);
        return false;
      }

      std::vector<Segment> segments(static_cast<size_t>(N));
      for (int s = 0; s < N; ++s) {
        segments[static_cast<size_t>(s)].T = T[static_cast<size_t>(s)];
        segments[static_cast<size_t>(s)].cx = sol_x.segment<kCoeffs>(kCoeffs * s);
        segments[static_cast<size_t>(s)].cy = sol_y.segment<kCoeffs>(kCoeffs * s);
      }

      int bad = -1;
      if (trajectoryClear(
          segments, grid_, clearance_, check_clearance, collision_dt_, bad))
      {
        publishResult(segments);
        const double ms = std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - t_start).count();
        RCLCPP_INFO(
          get_logger(),
          "Minimum snap 成功: %d 段, %zu 连接点, 离障 >= %.2f m, 用时 %.1f ms",
          N, waypoints.size(), check_clearance, ms);
        return true;
      }

      // 碰撞时优先延长时间，让多项式有更长的“余地”绕过去，
      // 而不是把连接点钉死或加 v=a=j=0 的停止约束。
      const double oldT = T[static_cast<size_t>(bad)];
      const double newT = std::min(max_seg_time_, oldT * 1.5 + 0.1);
      if (newT > oldT + 1e-6) {
        T[static_cast<size_t>(bad)] = newT;
        if (bad > 0) {
          T[static_cast<size_t>(bad - 1)] = std::min(
            max_seg_time_, T[static_cast<size_t>(bad - 1)] * 1.05 + 0.02);
        }
        if (bad + 1 < N) {
          T[static_cast<size_t>(bad + 1)] = std::min(
            max_seg_time_, T[static_cast<size_t>(bad + 1)] * 1.05 + 0.02);
        }
        continue;
      }

      // 时间已到上限：把碰撞段两端的连接点往更空旷的方向推一点
      if (nudgeWaypoints(waypoints, grid_, clearance_, bad, check_clearance)) {
        continue;
      }
      break;
    }

    return false;
  }

  // --------------------------------------------------------------------------
  // 发布
  // --------------------------------------------------------------------------
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

    double t0 = 0.0;
    double last_yaw = 0.0;

    for (size_t s = 0; s < segments.size(); ++s) {
      const Segment & seg = segments[s];
      const int samples = std::max(
        1, static_cast<int>(std::ceil(seg.T / std::max(0.005, sample_dt_))));
      for (int i = 0; i <= samples; ++i) {
        if (s > 0 && i == 0) {
          continue;
        }
        const double t = seg.T * static_cast<double>(i) / static_cast<double>(samples);
        const double tau = t / seg.T;

        const double px = evalPoly(seg.cx, 0, tau, seg.T);
        const double py = evalPoly(seg.cy, 0, tau, seg.T);
        const double vx = evalPoly(seg.cx, 1, tau, seg.T);
        const double vy = evalPoly(seg.cy, 1, tau, seg.T);
        const double ax = evalPoly(seg.cx, 2, tau, seg.T);
        const double ay = evalPoly(seg.cy, 2, tau, seg.T);

        if (std::hypot(vx, vy) > 1e-4) {
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

  // --------------------------------------------------------------------------
  // 成员
  // --------------------------------------------------------------------------
  rclcpp::Subscription<nav_msgs::msg::OccupancyGrid>::SharedPtr map_sub_;
  rclcpp::Subscription<nav_msgs::msg::Path>::SharedPtr path_sub_;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr pub_path_;
  rclcpp::Publisher<trajectory_msgs::msg::MultiDOFJointTrajectory>::SharedPtr pub_traj_;

  GridMap grid_;
  std::vector<float> clearance_;
  nav_msgs::msg::Path::SharedPtr last_path_;
  bool have_path_ = false;

  // 参数
  int occupied_thresh_ = 50;
  double hard_clearance_ = 0.35;
  double clearance_target_ = 1.5;
  double clearance_weight_ = 4.0;
  double guide_margin_ = 0.40;
  double waypoint_spacing_ = 1.2;
  double v_ref_ = 1.0;
  double min_seg_time_ = 1.0;
  double max_seg_time_ = 4.0;
  double sample_dt_ = 0.02;
  double collision_dt_ = 0.01;
  int max_iters_ = 30;
};

}  // namespace

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<MinSnapPlanner>());
  rclcpp::shutdown();
  return 0;
}
