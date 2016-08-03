#ifndef CONVERGENCECRITERIONHEADERDEF
#define CONVERGENCECRITERIONHEADERDEF

class ConvergenceCriterion
{

  public:

    // Specialised Constructor
    ConvergenceCriterion(const double tolerance);

    // Method for checking convergence based on the residual norm
    bool TestConvergence(const double residualNorm) const;

    // Accessor method
    void SetTolerance(const double tolerance);

  private:

    // Hidden default constructor
    ConvergenceCriterion();

    // Tolerance
    double mTolerance;

};

#endif

