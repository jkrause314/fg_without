
using namespace cv;

/*
 GMM - Gaussian Mixture Model
*/
class GMM
{
public:
  static const int componentsCount = 10;

  GMM( Mat& _model );
  double operator()( const Vec3d color ) const;
  double operator()( int ci, const Vec3d color ) const;
  int whichComponent( const Vec3d color ) const;

  void initLearning();
  void addSample( int ci, const Vec3d color );
  void addSample( int ci, const Vec3d color, float weight);
  void endLearning();

private:
  void calcInverseCovAndDeterm( int ci );
  Mat model;
  double* coefs;
  double* mean;
  double* cov;

  double inverseCovs[componentsCount][3][3];
  double covDeterms[componentsCount];

  double sums[componentsCount][3];
  double prods[componentsCount][3][3];
  double sampleCounts[componentsCount];
  double totalSampleCount;
};

GMM::GMM( Mat& _model )
{
  const int modelSize = 3/*mean*/ + 9/*covariance*/ + 1/*component weight*/;
  if( _model.empty() )
  {
    _model.create( 1, modelSize*componentsCount, CV_64FC1 );
    _model.setTo(Scalar(0));
  }
  else if( (_model.type() != CV_64FC1) || (_model.rows != 1) || (_model.cols != modelSize*componentsCount) )
    CV_Error( CV_StsBadArg, "_model must have CV_64FC1 type, rows == 1 and cols == 13*componentsCount" );

  model = _model;

  coefs = model.ptr<double>(0);
  mean = coefs + componentsCount;
  cov = mean + 3*componentsCount;

  for( int ci = 0; ci < componentsCount; ci++ )
    if( coefs[ci] > 0 )
       calcInverseCovAndDeterm( ci );
}

double GMM::operator()( const Vec3d color ) const
{
  double res = 0;
  for( int ci = 0; ci < componentsCount; ci++ )
    res += coefs[ci] * (*this)(ci, color );
  return res;
}

double GMM::operator()( int ci, const Vec3d color ) const
{
  double res = 0;
  if( coefs[ci] > 0 )
  {
    CV_Assert( covDeterms[ci] > std::numeric_limits<double>::epsilon() );
    Vec3d diff = color;
    double* m = mean + 3*ci;
    diff[0] -= m[0]; diff[1] -= m[1]; diff[2] -= m[2];
    double mult = diff[0]*(diff[0]*inverseCovs[ci][0][0] + diff[1]*inverseCovs[ci][1][0] + diff[2]*inverseCovs[ci][2][0])
           + diff[1]*(diff[0]*inverseCovs[ci][0][1] + diff[1]*inverseCovs[ci][1][1] + diff[2]*inverseCovs[ci][2][1])
           + diff[2]*(diff[0]*inverseCovs[ci][0][2] + diff[1]*inverseCovs[ci][1][2] + diff[2]*inverseCovs[ci][2][2]);
    res = 1.0f/sqrt(covDeterms[ci]) * exp(-0.5f*mult);
  }
  return res;
}

int GMM::whichComponent( const Vec3d color ) const
{
  int k = 0;
  double max = 0;

  for( int ci = 0; ci < componentsCount; ci++ )
  {
    double p = (*this)( ci, color );
    if( p > max )
    {
      k = ci;
      max = p;
    }
  }
  return k;
}

void GMM::initLearning()
{
  for( int ci = 0; ci < componentsCount; ci++)
  {
    sums[ci][0] = sums[ci][1] = sums[ci][2] = 0;
    prods[ci][0][0] = prods[ci][0][1] = prods[ci][0][2] = 0;
    prods[ci][1][0] = prods[ci][1][1] = prods[ci][1][2] = 0;
    prods[ci][2][0] = prods[ci][2][1] = prods[ci][2][2] = 0;
    sampleCounts[ci] = 0;
  }
  totalSampleCount = 0;
}

void GMM::addSample( int ci, const Vec3d color )
{
  sums[ci][0] += color[0]; sums[ci][1] += color[1]; sums[ci][2] += color[2];
  prods[ci][0][0] += color[0]*color[0]; prods[ci][0][1] += color[0]*color[1]; prods[ci][0][2] += color[0]*color[2];
  prods[ci][1][0] += color[1]*color[0]; prods[ci][1][1] += color[1]*color[1]; prods[ci][1][2] += color[1]*color[2];
  prods[ci][2][0] += color[2]*color[0]; prods[ci][2][1] += color[2]*color[1]; prods[ci][2][2] += color[2]*color[2];
  sampleCounts[ci]++;
  totalSampleCount++;
}


void GMM::addSample( int ci, const Vec3d color, float weight)
{
  sums[ci][0] += weight * color[0];
  sums[ci][1] += weight * color[1];
  sums[ci][2] += weight * color[2];
  prods[ci][0][0] += weight * color[0]*color[0];
  prods[ci][0][1] += weight * color[0]*color[1];
  prods[ci][0][2] += weight * color[0]*color[2];
  prods[ci][1][0] += weight * color[1]*color[0];
  prods[ci][1][1] += weight * color[1]*color[1];
  prods[ci][1][2] += weight * color[1]*color[2];
  prods[ci][2][0] += weight * color[2]*color[0];
  prods[ci][2][1] += weight * color[2]*color[1];
  prods[ci][2][2] += weight * color[2]*color[2];
  sampleCounts[ci] = sampleCounts[ci] + weight;
  totalSampleCount = totalSampleCount + weight;
}

void GMM::endLearning()
{
  const double variance = 0.01;
  for( int ci = 0; ci < componentsCount; ci++ )
  {
    double n = sampleCounts[ci];
    if( n == 0 )
      coefs[ci] = 0;
    else
    {
      coefs[ci] = (double)n/totalSampleCount;

      double* m = mean + 3*ci;
      m[0] = sums[ci][0]/n; m[1] = sums[ci][1]/n; m[2] = sums[ci][2]/n;

      double* c = cov + 9*ci;
      c[0] = prods[ci][0][0]/n - m[0]*m[0]; c[1] = prods[ci][0][1]/n - m[0]*m[1]; c[2] = prods[ci][0][2]/n - m[0]*m[2];
      c[3] = prods[ci][1][0]/n - m[1]*m[0]; c[4] = prods[ci][1][1]/n - m[1]*m[1]; c[5] = prods[ci][1][2]/n - m[1]*m[2];
      c[6] = prods[ci][2][0]/n - m[2]*m[0]; c[7] = prods[ci][2][1]/n - m[2]*m[1]; c[8] = prods[ci][2][2]/n - m[2]*m[2];

      double dtrm = c[0]*(c[4]*c[8]-c[5]*c[7]) - c[1]*(c[3]*c[8]-c[5]*c[6]) + c[2]*(c[3]*c[7]-c[4]*c[6]);
      if( dtrm <= std::numeric_limits<double>::epsilon() )
      {
        // Adds the white noise to avoid singular covariance matrix.
        c[0] += variance;
        c[4] += variance;
        c[8] += variance;
      }

      calcInverseCovAndDeterm(ci);
    }
  }
}

void GMM::calcInverseCovAndDeterm( int ci )
{
  if( coefs[ci] > 0 )
  {
    double *c = cov + 9*ci;
    double dtrm =
        covDeterms[ci] = c[0]*(c[4]*c[8]-c[5]*c[7]) - c[1]*(c[3]*c[8]-c[5]*c[6]) + c[2]*(c[3]*c[7]-c[4]*c[6]);

    CV_Assert( dtrm > std::numeric_limits<double>::epsilon() );
    inverseCovs[ci][0][0] =  (c[4]*c[8] - c[5]*c[7]) / dtrm;
    inverseCovs[ci][1][0] = -(c[3]*c[8] - c[5]*c[6]) / dtrm;
    inverseCovs[ci][2][0] =  (c[3]*c[7] - c[4]*c[6]) / dtrm;
    inverseCovs[ci][0][1] = -(c[1]*c[8] - c[2]*c[7]) / dtrm;
    inverseCovs[ci][1][1] =  (c[0]*c[8] - c[2]*c[6]) / dtrm;
    inverseCovs[ci][2][1] = -(c[0]*c[7] - c[1]*c[6]) / dtrm;
    inverseCovs[ci][0][2] =  (c[1]*c[5] - c[2]*c[4]) / dtrm;
    inverseCovs[ci][1][2] = -(c[0]*c[5] - c[2]*c[3]) / dtrm;
    inverseCovs[ci][2][2] =  (c[0]*c[4] - c[1]*c[3]) / dtrm;
  }
}
