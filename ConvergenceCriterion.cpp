#include"ConvergenceCriterion.hpp"

// Constructor
ConvergenceCriterion::
ConvergenceCriterion(const double tolerance)
{
  mTolerance = tolerance;
}

// True if || r || < tolerance
bool ConvergenceCriterion::
TestConvergence(const double residualNorm) const
{
  return residualNorm <= mTolerance;
}


void ConvergenceCriterion::
SetTolerance(const double tolerance)
{
  mTolerance = tolerance;
}
