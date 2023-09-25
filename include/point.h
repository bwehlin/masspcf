namespace mpcf
{
  template <typename Tt, typename Tv>
  struct Point
  {
    using time_type = Tt;
    using value_type = Tv;

    Tt t = 0.; // time
    Tv v = 0.; // value

    Point() = default;
    Point(Tt it, Tv iv)
      : t(it), v(iv) { }
  };
}
