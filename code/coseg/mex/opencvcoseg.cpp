/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//            Intel License Agreement
//        For Open Source Computer Vision Library
//
// Copyright (C) 2000, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//   this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//   this list of conditions and the following disclaimer in the documentation
//   and/or other materials provided with the distribution.
//
//   * The name of Intel Corporation may not be used to endorse or promote products
//   derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#include "gcgraph.hpp"
#include "gmm.cpp"
#include "coseg_options.hpp"
#include "opencv2/opencv.hpp"
#include <limits>
#include <algorithm>
#include <math.h>

using namespace cv;

// Mean + std
static float adaptiveThresh(const Mat& sal) {
  Scalar mean;
  Scalar stddev;
  meanStdDev(sal, mean, stddev);
  double thresh = mean[0] + .5 * stddev[0];
  int min_fg = 255 * sal.cols * sal.rows * .001;
  Mat in_fg = sal > thresh;
  while (sum(in_fg)[0] <= min_fg) {
    thresh /= 1.2;
    in_fg = sal > thresh;
  }
  return thresh;
}


/*
  Calculate beta - parameter of GrabCut algorithm.
  beta = 1/(2*avg(sqr(||color[i] - color[j]||)))
*/
static double calcBeta( const Mat& img ) {
  double beta = 0;
  for( int y = 0; y < img.rows; y++ ) {
    for( int x = 0; x < img.cols; x++ ) {
      Vec3d color = img.at<Vec3b>(y,x);
      if( x>0 ) { // left
        Vec3d diff = color - (Vec3d)img.at<Vec3b>(y,x-1);
        beta += diff.dot(diff);
      }
      if( y>0 && x>0 ) { // upleft
        Vec3d diff = color - (Vec3d)img.at<Vec3b>(y-1,x-1);
        beta += diff.dot(diff);
      }
      if( y>0 ) { // up
        Vec3d diff = color - (Vec3d)img.at<Vec3b>(y-1,x);
        beta += diff.dot(diff);
      }
      if( y>0 && x<img.cols-1) { // upright
        Vec3d diff = color - (Vec3d)img.at<Vec3b>(y-1,x+1);
        beta += diff.dot(diff);
      }
    }
  }
  if( beta <= std::numeric_limits<double>::epsilon() )
    beta = 0;
  else
    beta = 1.f / (2 * beta/(4*img.cols*img.rows - 3*img.cols - 3*img.rows + 2) );

  return beta;
}

/*
  Calculate weights of noterminal vertices of graph.
  beta and gamma - parameters of GrabCut algorithm.
 */
static void calcNWeights( const Mat& img, Mat& leftW, Mat& upleftW, Mat& upW, Mat& uprightW, double beta, double gamma )
{
  const double gammaDivSqrt2 = gamma / std::sqrt(2.0f);
  leftW.create( img.rows, img.cols, CV_64FC1 );
  upleftW.create( img.rows, img.cols, CV_64FC1 );
  upW.create( img.rows, img.cols, CV_64FC1 );
  uprightW.create( img.rows, img.cols, CV_64FC1 );
  for( int y = 0; y < img.rows; y++ ) {
    for( int x = 0; x < img.cols; x++ ) {
      Vec3d color = img.at<Vec3b>(y,x);
      if( x-1>=0 ) { // left
        Vec3d diff = color - (Vec3d)img.at<Vec3b>(y,x-1);
        leftW.at<double>(y,x) = gamma * exp(-beta*diff.dot(diff));
      } else {
        leftW.at<double>(y,x) = 0;
      }
      if( x-1>=0 && y-1>=0 ) { // upleft
        Vec3d diff = color - (Vec3d)img.at<Vec3b>(y-1,x-1);
        upleftW.at<double>(y,x) = gammaDivSqrt2 * exp(-beta*diff.dot(diff));
      } else {
        upleftW.at<double>(y,x) = 0;
      }
      if( y-1>=0 ) { // up
        Vec3d diff = color - (Vec3d)img.at<Vec3b>(y-1,x);
        upW.at<double>(y,x) = gamma * exp(-beta*diff.dot(diff));
      } else {
        upW.at<double>(y,x) = 0;
      }
      if( x+1<img.cols && y-1>=0 ) { // upright
        Vec3d diff = color - (Vec3d)img.at<Vec3b>(y-1,x+1);
        uprightW.at<double>(y,x) = gammaDivSqrt2 * exp(-beta*diff.dot(diff));
      } else {
        uprightW.at<double>(y,x) = 0;
      }
    }
  }
}


static void initGMMKmeans(std::vector<Vec3f>& samples, GMM& gmm) {
  Mat labels;
  cv::theRNG().state = 0;
  const int kMeansItCount = 10;
  const int kMeansType = KMEANS_PP_CENTERS;

  CV_Assert( !samples.empty() );
  Mat _samples( (int)samples.size(), 3, CV_32FC1, &samples[0][0] );
  kmeans( _samples, GMM::componentsCount, labels,
      TermCriteria( CV_TERMCRIT_ITER, kMeansItCount, 0.0), 0, kMeansType );

  gmm.initLearning();
  for( int i = 0; i < (int)samples.size(); i++ )
    gmm.addSample( labels.at<int>(i,0), samples[i] );
  gmm.endLearning();
}

static void initGMMKmeans(std::vector<Vec3f>& bgdSamples, std::vector<Vec3f>& fgdSamples, GMM& bgdGMM, GMM& fgdGMM) {
  initGMMKmeans(bgdSamples, bgdGMM);
  initGMMKmeans(fgdSamples, fgdGMM);
}


/*
  Construct GCGraph
*/
static void constructGCGraph( const Mat& img, const Mat& mask,
    const GMM& bgdGMM, const GMM& fgdGMM, const double lambda,
    const Mat& leftW, const Mat& upleftW, const Mat& upW, const Mat& uprightW,
    const GMM& bgdGMM_class, const GMM& fgdGMM_class, coseg_options options,
    GCGraph<double>& graph ) {

  double class_weight = options.class_weight;
  bool use_class_fg = options.use_class_fg;
  bool use_class_bg = options.use_class_bg;
  double fg_prior = options.fg_prior;

  int vtxCount = img.cols*img.rows,
    edgeCount = 2*(4*img.cols*img.rows - 3*(img.cols + img.rows) + 2);
  graph.create(vtxCount, edgeCount);
  Point p;
  for( p.y = 0; p.y < img.rows; p.y++ ) {
    for( p.x = 0; p.x < img.cols; p.x++) {
      // add node
      int vtxIdx = graph.addVtx();
      Vec3b color = img.at<Vec3b>(p);

      // set t-weights
      double fromSource, toSink;
      if( mask.at<uchar>(p) == GC_PR_BGD || mask.at<uchar>(p) == GC_PR_FGD ) {
        // add to fromSource to make foreground more likely
        // add to toSink to make foreground less likely
        if (use_class_fg)
          toSink = (1 - class_weight) * -log(fgdGMM(color)) - class_weight * log(fgdGMM_class(color));
        else
          toSink = -log(fgdGMM(color));
        if (use_class_bg) {
          fromSource = (1 - class_weight) * -log(bgdGMM(color)) - class_weight * log(bgdGMM_class(color));
        } else {
          fromSource = -log(bgdGMM(color));
        }
        fromSource -= log(1 - fg_prior);
        toSink -= log(fg_prior);
      }
      else if( mask.at<uchar>(p) == GC_BGD ) {
        fromSource = 0;
        toSink = lambda;
      } else { // GC_FGD
        printf("NO FGD\n");
        fromSource = lambda;
        toSink = 0;
      }
      graph.addTermWeights( vtxIdx, fromSource, toSink );

      // set n-weights
      if( p.x>0 ) {
        double w = leftW.at<double>(p);
        graph.addEdges( vtxIdx, vtxIdx-1, w, w );
      }
      if( p.x>0 && p.y>0 ) {
        double w = upleftW.at<double>(p);
        graph.addEdges( vtxIdx, vtxIdx-img.cols-1, w, w );
      }
      if( p.y>0 ) {
        double w = upW.at<double>(p);
        graph.addEdges( vtxIdx, vtxIdx-img.cols, w, w );
      }
      if( p.x<img.cols-1 && p.y>0 ) {
        double w = uprightW.at<double>(p);
        graph.addEdges( vtxIdx, vtxIdx-img.cols+1, w, w );
      }
    }
  }
}


// Initialize lots of stuff.
static void initAll(const vector<Mat>& imgs,
    const vector<Mat>& gt_masks,
    vector<Mat>& masks,
    vector<Mat>& compIdxs_im,
    vector<Mat>& compIdxs_class_im,
    vector<GMM>& bgdGMM_im,
    vector<GMM>& fgdGMM_im,
    GMM& bgdGMM_class,
    GMM& fgdGMM_class,
    vector<Mat>& leftW_im,
    vector<Mat>& upleftW_im,
    vector<Mat>& upW_im,
    vector<Mat>& uprightW_im,
    vector<double>& lambda_im,
    const coseg_options options) {

  vector<double> gamma_im;
  vector<double> beta_im;

  bool do_coseg = options.use_class_fg || options.use_class_bg;

  for (int i = 0; i < imgs.size(); ++i) {
    Mat bgd_model;
    Mat fgd_model;
    bgdGMM_im.push_back(GMM(bgd_model));
    fgdGMM_im.push_back(GMM(fgd_model));
    compIdxs_im.push_back(Mat(imgs[i].size(), CV_32SC1));
    gamma_im.push_back(50);
    lambda_im.push_back(9*gamma_im[i]);
    beta_im.push_back(calcBeta(imgs[i]));
    leftW_im.push_back(Mat());
    upleftW_im.push_back(Mat());
    upW_im.push_back(Mat());
    uprightW_im.push_back(Mat());
    calcNWeights(imgs[i], leftW_im[i], upleftW_im[i], upW_im[i], uprightW_im[i], beta_im[i], gamma_im[i]);
    if (do_coseg)
      compIdxs_class_im.push_back(Mat(imgs[i].size(), CV_32SC1));
  }

  // GMMs and masks together
  std::vector<Vec3f> bgdSamples_class, fgdSamples_class;
  for (int i = 0; i < imgs.size(); ++i) {
    const Mat& img = imgs[i];
    const Mat& gt_mask = gt_masks[i];
    Mat& mask = masks[i];
    mask.create(img.size(), CV_8UC1);
    mask.setTo(GC_PR_BGD);

    Point p;

    std::vector<Vec3f> bgdSamples, fgdSamples;
    for( p.y = 0; p.y < img.rows; p.y++ ) {
      for( p.x = 0; p.x < img.cols; p.x++ ) {
        const Vec3f& sample = (Vec3f)img.at<Vec3b>(p);

        // Transfer gt over, otherwise compare to thresh
        if (gt_mask.at<uchar>(p) == GC_BGD) {
          //printf("I got background :( :( :(\n");
          bgdSamples.push_back(sample);
          if (options.use_class_bg)
            bgdSamples_class.push_back(sample);
          mask.at<uchar>(p) = GC_BGD;
        } else if (gt_mask.at<uchar>(p) == GC_FGD) {
          //printf("I got foreground!\n");
          fgdSamples.push_back(sample);
          if (options.use_class_fg)
            fgdSamples_class.push_back(sample);
          mask.at<uchar>(p) = GC_FGD;
        } else if (gt_mask.at<uchar>(p) == GC_PR_BGD) {
          bgdSamples.push_back(sample);
          if (options.use_class_bg)
            bgdSamples_class.push_back(sample);
          mask.at<uchar>(p) = GC_PR_BGD;
        } else if (gt_mask.at<uchar>(p) == GC_PR_FGD) {
          fgdSamples.push_back(sample);
          if (options.use_class_fg)
            fgdSamples_class.push_back(sample);
          mask.at<uchar>(p) = GC_PR_FGD;
        } else {
          assert(false);
        }
      }
    }
    initGMMKmeans(bgdSamples, fgdSamples, bgdGMM_im[i], fgdGMM_im[i]);
  }
  if (options.use_class_bg) {
    initGMMKmeans(bgdSamples_class, bgdGMM_class);
  }
  if (options.use_class_fg) {
    initGMMKmeans(fgdSamples_class, fgdGMM_class);
  }
}


// Do an iteration of GMM learning
static void iterGMM_single(
    const Mat& img,
    const Mat& mask,
    GMM& bgdGMM,
    GMM& fgdGMM,
    Mat& compIdxs) {

  Point p;
  for( p.y = 0; p.y < img.rows; p.y++ ) {
    for( p.x = 0; p.x < img.cols; p.x++ ) {
      const Vec3d& color = img.at<Vec3b>(p);
      bool bg = mask.at<uchar>(p) == GC_BGD || mask.at<uchar>(p) == GC_PR_BGD;
      if (bg) {
        compIdxs.at<int>(p) = bgdGMM.whichComponent(color);
      } else {
        compIdxs.at<int>(p) = fgdGMM.whichComponent(color);
      }
    }
  }
  bgdGMM.initLearning();
  fgdGMM.initLearning();
  for( p.y = 0; p.y < img.rows; p.y++ ) {
    for( p.x = 0; p.x < img.cols; p.x++ ) {
      const Vec3b& color = img.at<Vec3b>(p);
      if( mask.at<uchar>(p) == GC_BGD || mask.at<uchar>(p) == GC_PR_BGD ) {
        bgdGMM.addSample(compIdxs.at<int>(p), color);
      } else {
        fgdGMM.addSample(compIdxs.at<int>(p), color);
      }
    }
  }
  bgdGMM.endLearning();
  fgdGMM.endLearning();
}


// Do an iteration of GMM learning
static void iterGMMs(
    const vector<Mat>& imgs,
    const vector<Mat>& masks,
    vector<GMM>& bgdGMM_im,
    vector<GMM>& fgdGMM_im,
    vector<Mat>& compIdxs_im,
    GMM& bgdGMM_class,
    GMM& fgdGMM_class,
    vector<Mat>& compIdxs_class_im,
    const coseg_options options) {

  // Set component indices
  for (int i = 0; i < imgs.size(); ++i) {
    const Mat& img = imgs[i];
    GMM& fgdGMM = fgdGMM_im[i];
    GMM& bgdGMM = bgdGMM_im[i];
    const Mat& mask = masks[i];
    Mat& compIdxs = compIdxs_im[i];
    Mat& compIdxs_class = compIdxs_class_im[i];
    Point p;
    for( p.y = 0; p.y < img.rows; p.y++ ) {
      for( p.x = 0; p.x < img.cols; p.x++ ) {
        const Vec3d& color = img.at<Vec3b>(p);
        bool bg = mask.at<uchar>(p) == GC_BGD || mask.at<uchar>(p) == GC_PR_BGD;
        if (bg) {
          compIdxs.at<int>(p) = bgdGMM.whichComponent(color);
          if (options.use_class_bg)
            compIdxs_class.at<int>(p) = bgdGMM_class.whichComponent(color);
        } else {
          compIdxs.at<int>(p) = fgdGMM.whichComponent(color);
          if (options.use_class_fg)
            compIdxs_class.at<int>(p) = fgdGMM_class.whichComponent(color);
        }
      }
    }
  }

  // Learn GMMs
  if (options.use_class_bg) {
    bgdGMM_class.initLearning();
  }
  if (options.use_class_fg) {
    fgdGMM_class.initLearning();
  }
  for (int i = 0; i < imgs.size(); ++i) {
    const Mat& img = imgs[i];
    GMM& fgdGMM = fgdGMM_im[i];
    GMM& bgdGMM = bgdGMM_im[i];
    const Mat& mask = masks[i];
    Mat& compIdxs = compIdxs_im[i];
    Mat& compIdxs_class = compIdxs_class_im[i];

    bgdGMM.initLearning();
    fgdGMM.initLearning();
    Point p;
    for( p.y = 0; p.y < img.rows; p.y++ ) {
      for( p.x = 0; p.x < img.cols; p.x++ ) {
        const Vec3b& color = img.at<Vec3b>(p);
        if( mask.at<uchar>(p) == GC_BGD || mask.at<uchar>(p) == GC_PR_BGD ) {
          bgdGMM.addSample(compIdxs.at<int>(p), color);
          if (options.use_class_bg)
            bgdGMM_class.addSample(compIdxs_class.at<int>(p), color);
        } else {
          fgdGMM.addSample(compIdxs.at<int>(p), color);
          if (options.use_class_fg)
            fgdGMM_class.addSample(compIdxs_class.at<int>(p), color);
        }
      }
    }
    bgdGMM.endLearning();
    fgdGMM.endLearning();
  }
  if (options.use_class_bg) {
    bgdGMM_class.endLearning();
  }
  if (options.use_class_fg) {
    fgdGMM_class.endLearning();
  }
}

/*
  Estimate segmentation using MaxFlow algorithm
*/
static void estimateSegmentation( GCGraph<double>& graph, Mat& mask ) {
  graph.maxFlow();
  Point p;
  for( p.y = 0; p.y < mask.rows; p.y++ ) {
    for( p.x = 0; p.x < mask.cols; p.x++ ) {
      if( mask.at<uchar>(p) == GC_PR_BGD || mask.at<uchar>(p) == GC_PR_FGD ) {
        if( graph.inSourceSegment( p.y*mask.cols+p.x /*vertex index*/ ) )
          mask.at<uchar>(p) = GC_PR_FGD;
        else
          mask.at<uchar>(p) = GC_PR_BGD;
      }
    }
  }
}

void validateInput(const vector<Mat>& imgs,  const vector<Mat>& masks, const coseg_options options) {
  if (imgs.size() != masks.size())
    CV_Error(CV_StsBadArg, "images, saliencies, masks must be same size");
  for (int i = 0; i < imgs.size(); ++i) {
    if( imgs[i].empty() )
      CV_Error( CV_StsBadArg, "image is empty" );
    if( imgs[i].type() != CV_8UC3 )
      CV_Error( CV_StsBadArg, "image must have CV_8UC3 type" );
  }
}

static void refineSeg( const Mat& img, Mat& mask,
    GMM& bgdGMM, GMM& fgdGMM, Mat& compIdxs, const double lambda,
    const Mat& leftW, const Mat& upleftW, const Mat& upW, const Mat& uprightW,
    const double min_fg_area, const double max_fg_area,
    const double min_fg_length, const double min_fg_height,
    const GMM& bgdGMM_class, const GMM& fgdGMM_class, coseg_options options) {
  // Check if satisfies criterion
  // If so, stop
  // If not, update the fg prior
  double old_fg_prior = options.fg_prior;
  int max_refine_bs_iters = 20;
  double lower_prior = 0;
  double upper_prior = 1;
  for (int i = 0; i < max_refine_bs_iters; ++i) {
    // Check if satisfies criterion
    int min_x = -1;
    int max_x = -1;
    int min_y = -1;
    int max_y = -1;
    int area = 0;
    Point p;
    for( p.y = 0; p.y < img.rows; p.y++ ) {
      for( p.x = 0; p.x < img.cols; p.x++) {
        if (mask.at<uchar>(p) == GC_PR_FGD) {
          min_x = min_x == -1 ? p.x : std::min(min_x, p.x);
          max_x = max_x == -1 ? p.x : std::max(max_x, p.x);
          min_y = min_y == -1 ? p.y : std::min(min_y, p.y);
          max_y = max_y == -1 ? p.y : std::max(max_y, p.y);
          area++;
        }
      }
    }
    bool too_small = max_x-min_x+1 < min_fg_length ||
      max_y-min_y+1 < min_fg_height ||
      area < min_fg_area;
    bool too_large = area > max_fg_area;
    if (!too_small && !too_large) {
      break;
    }
    double old_prior = (lower_prior + upper_prior) / 2;
    if (too_small) {
      lower_prior = old_prior;
    } else {
      upper_prior = old_prior;
    }
    double new_prior = (lower_prior + upper_prior) / 2;
    options.fg_prior = new_prior;

    // Reset mask
    for( p.y = 0; p.y < img.rows; p.y++ ) {
      for( p.x = 0; p.x < img.cols; p.x++) {
        if (mask.at<uchar>(p) != GC_BGD) {
          mask.at<uchar>(p) = GC_PR_FGD;
        }
      }
    }
    for (int j = 0; j < options.iterCount; ++j) {
      // Update GMMs
      iterGMM_single(img, mask, bgdGMM, fgdGMM, compIdxs);
      // Update segmentation
      GCGraph<double> graph;
      constructGCGraph(img, mask, bgdGMM, fgdGMM, lambda, leftW, upleftW, upW, uprightW, bgdGMM_class, fgdGMM_class, options, graph);
      estimateSegmentation(graph, mask);
    }
  }
  options.fg_prior = old_fg_prior;
}


// Main function
void runCoseg(
    const vector<Mat>& imgs,
    const vector<Mat>& gt_masks,
    vector<Mat>& masks,
    const vector<double>& min_fg_areas,
    const vector<double>& max_fg_areas,
    const vector<double>& min_fg_lengths,
    const vector<double>& min_fg_heights,
    const coseg_options options) {

  int iterCount = options.iterCount;
  validateInput(imgs, masks, options);

  // Initialize everything
  // Image-specific
  vector<Mat> bgdModel_im;
  vector<Mat> fgdModel_im;
  vector<GMM> bgdGMM_im;
  vector<GMM> fgdGMM_im;
  vector<Mat> compIdxs_im;
  vector<Mat> leftW_im;
  vector<Mat> upleftW_im;
  vector<Mat> upW_im;
  vector<Mat> uprightW_im;
  vector<double> lambda_im;
  vector<double> thresh_im;
  // Class-level
  Mat bgdModel_class;
  Mat fgdModel_class;
  GMM bgdGMM_class(bgdModel_class);
  GMM fgdGMM_class(fgdModel_class);
  vector<Mat> compIdxs_class_im;

  initAll(imgs, gt_masks, masks,
      compIdxs_im, compIdxs_class_im,
      bgdGMM_im, fgdGMM_im, bgdGMM_class, fgdGMM_class,
      leftW_im, upleftW_im, upW_im, uprightW_im, lambda_im, options);

  // Iterate
  for (int j = 0; j < iterCount; j++) {
    printf("coseg iter %d/%d\n", j+1, iterCount);

    iterGMMs(imgs, masks, bgdGMM_im, fgdGMM_im, compIdxs_im, bgdGMM_class, fgdGMM_class, compIdxs_class_im, options);

    for (int i = 0; i < imgs.size(); ++i) {
      GCGraph<double> graph;
      constructGCGraph(imgs[i], masks[i], bgdGMM_im[i], fgdGMM_im[i], lambda_im[i], leftW_im[i], upleftW_im[i], upW_im[i], uprightW_im[i], bgdGMM_class, fgdGMM_class, options, graph);
      estimateSegmentation(graph, masks[i]);
    }
  }
  if (options.do_refine) {
    for (int i = 0; i < imgs.size(); ++i) {
      //printf("refine %d\n", i);
      refineSeg(imgs[i], masks[i],
          bgdGMM_im[i], fgdGMM_im[i], compIdxs_im[i], lambda_im[i],
          leftW_im[i], upleftW_im[i], upW_im[i], uprightW_im[i],
          min_fg_areas[i], max_fg_areas[i], min_fg_lengths[i], min_fg_heights[i],
          bgdGMM_class, fgdGMM_class, options);
    }
  }
}
