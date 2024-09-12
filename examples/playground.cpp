#include "unscented/primitives.h"
#include "unscented/ukf.h"

#include <iostream>

int main()
{
  using MState = unscented::Compound<unscented::Vector<2>, unscented::Angle>;
  MState t;
  t.data =
      std::make_tuple(unscented::Vector<2>{-0.1, 0.2}, unscented::Angle{0.2});
  std::cout << std::get<0>(t.data).transpose() << " "
            << " a: " << std::get<1>(t.data).get_angle() << "\n";
  auto x = operator+(t, unscented::Vector<3>{1.0, 2.0, 3.0});
  // auto x = t + unscented::Vector<3>{1.0, 2.0, 3.0};
  std::cout << std::get<0>(x.data).transpose() << " "
            << " a: " << std::get<1>(x.data).get_angle() << "\n";
  auto v = x - t;
  std::cout << "v: " << v.transpose() << "\n";

  return 0;
}
