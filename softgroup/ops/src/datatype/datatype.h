#ifndef DATATYPE_H
#define DATATYPE_H
#include <array>
#include <cstdint>
#include <google/dense_hash_map>
#include <queue>
#include <vector>

using Int = int32_t;

template <Int dimension> using Point = std::array<Int, dimension>;

template <Int dimension> struct IntArrayHash {
  std::size_t operator()(Point<dimension> const &p) const {
    Int hash = 16777619;
    for (auto x : p) {
      hash *= 2166136261;
      hash ^= x;
    }
    return hash;
  }
};

template <Int dimension>
using SparseGridMap = google::dense_hash_map<
    Point<dimension>, Int, IntArrayHash<dimension>,
    std::equal_to<Point<dimension>>>; // <Key, Data, HashFcn, EqualKey, Alloc>

template <Int dimension> class SparseGrid {
public:
  Int ctr;
  SparseGridMap<dimension> mp;
  SparseGrid();
};

template <Int dimension> using SparseGrids = std::vector<SparseGrid<dimension>>;

using RuleBook = std::vector<std::vector<Int>>;

class ConnectedComponent {
public:
  std::vector<Int> pt_idxs;
  float accum_x = 0.;
  float accum_y = 0.;
  float accum_z = 0.;
  int cls_label = -100;
  int batch_idx = -1;
  // int npoint = 0;

  ConnectedComponent();
  void addPoint(Int pt_idx);
};

using ConnectedComponents = std::vector<ConnectedComponent>;

#endif // DATATYPE_H
