#include "unscented/primitives.hpp"
#include "unscented/ukf.hpp"

int main() {
  using UKF = unscented::UKF<unscented::Vector<2>, unscented::Vector<1>>;
  UKF ukf;
}
