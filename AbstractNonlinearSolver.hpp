#ifndef ABSTRACTNONLINEARSOLVERHEADERDEF
#define ABSTRACTNONLINEARSOLVERHEADERDEF

// #include "AbstractNonlinearProblemRequired.hpp"
// #include "AbstractNonlinearProblemJacobian.hpp"
// #include "ConvergenceCriterion.hpp"
#include <armadillo>
#include <string>

class AbstractNonlinearSolver
{

  public:

    // Convergence flag
    enum class ExitFlagType {
      converged,
      notConverged
    };

    // Solution method
    virtual void Solve(arma::vec& solution, 
	arma::vec& residualHistory,
	ExitFlagType& exitFlag,
  arma::mat* pJacobianExternal=NULL) = 0;

  protected:

    // Printing infos when iterations begin
    virtual void PrintHeader(const std::string solverName, 
	int maxIterations, double tolerance) const;

    // Printing infos when iterations terminate
    virtual void PrintFooter(const int iteration,
	const ExitFlagType exitFlag) const;

    // Print infos about current iterate
    virtual void PrintIteration(const int iteration, 
	const double errorEstimate,
	const bool initialise = false) const;

};


#endif
