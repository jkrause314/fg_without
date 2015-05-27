/**
 * @file grabCut.cpp
 * @brief mex interface for grabCut
 * @author Kota Yamaguchi
 * @date 2012
 */
#include "MxArray.hpp"
#include "coseg_options.hpp"
#include "opencvcoseg.cpp"
using namespace std;

/**
 * Main entry called from Matlab
 * @param nlhs number of left-hand-side arguments
 * @param plhs pointers to mxArrays in the left-hand-side
 * @param nrhs number of right-hand-side arguments
 * @param prhs pointers to mxArrays in the right-hand-side
 *
 * This is the entry point of the function
 */
void mexFunction( int nlhs, mxArray *plhs[],
          int nrhs, const mxArray *prhs[] )
{
  // Argument vector
  vector<MxArray> rhs(prhs,prhs+nrhs);

  // Option processing
  MxArray ims_arr = rhs[0];
  MxArray gt_mask_arr = rhs[1];
  MxArray min_fg_area_arr = rhs[2];
  MxArray max_fg_area_arr = rhs[3];
  MxArray min_fg_length_arr = rhs[4];
  MxArray min_fg_height_arr = rhs[5];
  coseg_options options;
  MxArray options_mat = rhs[6];

  if (options_mat.isField("num_iters"))
    options.iterCount = options_mat.at("num_iters").toInt();
  if (options_mat.isField("class_weight"))
    options.class_weight = options_mat.at("class_weight").toDouble();
  if (options_mat.isField("use_class_fg"))
    options.use_class_fg = options_mat.at("use_class_fg").toBool();
  if (options_mat.isField("use_class_bg"))
    options.use_class_bg = options_mat.at("use_class_bg").toBool();
  if (options_mat.isField("fg_prior"))
    options.fg_prior = options_mat.at("fg_prior").toDouble();
  if (options_mat.isField("do_refine"))
    options.do_refine = options_mat.at("do_refine").toBool();


  int num_ims = ims_arr.numel();

  vector<cv::Mat> imgs;
  vector<cv::Mat> gt_masks;
  vector<cv::Mat> masks;

  vector<double> min_fg_areas;
  vector<double> max_fg_areas;
  vector<double> min_fg_lengths;
  vector<double> min_fg_heights;

  // Load up the images
  for (int i = 0; i < num_ims; ++i) {
    mxArray *imarr = mxGetCell(ims_arr, i);
    if (mxGetClassID(imarr) != mxUINT8_CLASS)
      mexErrMsgIdAndTxt("mexopencv:error","Only UINT8 type is supported");
    if (mxGetNumberOfDimensions(imarr) != 3)
      mexErrMsgIdAndTxt("mexopencv:error","Only RGB format is supported");
    imgs.push_back(MxArray(imarr).toMat(CV_8U));
    gt_masks.push_back(MxArray(mxGetCell(gt_mask_arr, i)).toMat(CV_8U));
    masks.push_back(cv::Mat());
    min_fg_areas.push_back(min_fg_area_arr.at<double>(i));
    max_fg_areas.push_back(max_fg_area_arr.at<double>(i));
    min_fg_lengths.push_back(min_fg_length_arr.at<double>(i));
    min_fg_heights.push_back(min_fg_height_arr.at<double>(i));
  }

  runCoseg(imgs, gt_masks, masks, min_fg_areas, max_fg_areas, min_fg_lengths, min_fg_heights, options);

  // Write out the masks
  plhs[0] = mxCreateCellMatrix(1, num_ims);
  for (int i = 0; i < num_ims; ++i) {
    mxSetCell(plhs[0], i, MxArray(masks[i]));
  }
}
