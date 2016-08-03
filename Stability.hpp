#ifndef STABILITYHEADERDEF
#define STABILITYHEADERDEF

#include <cassert>
#include "AbstractNonlinearProblem.hpp"
#include "AbstractNonlinearProblemJacobian.hpp"

class Stability
{
  public:

    // Problem type
    enum class ProblemType
    {
      flow,
      map,
      equationFree
    };

    // Constructor (Jacobian computed by finite differences explicitly)
    Stability( ProblemType type, AbstractNonlinearProblem* pProblem);

    // Constructor (Jacobian passed by user)
    Stability( ProblemType type,
               AbstractNonlinearProblem* pProblem,
               AbstractNonlinearProblemJacobian *pProblemJacobian);

    ~Stability();

    int ComputeNumUnstableEigenvalues(const arma::vec& u);

    int ComputeNumUnstableEigenvalues(const arma::mat& jacobian);

  private:

    // Hiding default constructor
    Stability();

    arma::cx_vec ComputeEigenvalues(const arma::vec& u);

    // Compute Jacobian via finite differences
    void ComputeDFDU(const arma::vec& u, arma::mat& jacobian);

    // Problem interface
    AbstractNonlinearProblem* mpProblem;
    AbstractNonlinearProblemJacobian* mpProblemJacobian;
    ProblemType mProblemType;

    // Other parameters
    double mFiniteDifferenceEpsilon;

};

#endif
