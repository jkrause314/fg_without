#ifndef __COSEG_OPTIONS_HPP__
#define __COSEG_OPTIONS_HPP__

struct coseg_options {
  coseg_options() : iterCount(5), class_weight(.5), use_class_fg(true), use_class_bg(false), fg_prior(.5), do_refine(false) {}
  int iterCount;
  double class_weight;
  bool use_class_fg;
  bool use_class_bg;
  double fg_prior;
  bool do_refine;
};

#endif // __COSEG_OPTIONS_HPP__
