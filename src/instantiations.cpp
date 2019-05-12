#include "unscented.hpp"

namespace unscented
{
  template class UKF<Eigen::Matrix<double, 6, 1>, 6, Eigen::Vector3d, 3>;
} // namespace unscented
