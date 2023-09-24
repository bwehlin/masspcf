namespace mpcf
{
  template <typename Tt, typename Tv>
  struct Rectangle
  {
    Tt left = 0;
    Tt right = 0;
    Tv top = 0;
    Tv bottom = 0;

    Rectangle() = default;
    Rectangle(Tt l, Tt r, Tv t, Tv b)
      : left(l), right(r), top(t), bottom(b)
    { }
  };
}
