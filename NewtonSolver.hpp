#ifndef NEWTONSOLVERHEADERDEF
#define NEWTONSOLVERHEADERDEF

#include <armadillo>
#include "AbstractNonlinearProblem.hpp"
#include "AbstractNonlinearProblemJacobian.hpp"
#include "ConvergenceCriterion.hpp"
#include "AbstractNonlinearSolver.hpp"

class NewtonSolver:
  public AbstractNonlinearSolver
{

  public:

    // Parameter list
    struct ParameterList {

      ParameterList()
      {
	tolerance = 1e-5;
	maxIterations = 10;
	printOutput = true;
	finiteDifferenceEpsilon = 1e-8;
  damping = 1.0;
      };

      double tolerance;
      int maxIterations;
      bool printOutput;
      double finiteDifferenceEpsilon;
      double damping;

    };

    // Constructor (Jacobian computed by finite differences explicitly)
    NewtonSolver(
	AbstractNonlinearProblem* pProblem,
	const arma::vec* pInitialGuess,
	const ParameterList* pParameterList);

    // Constructor (Jacobian passed by user)
    NewtonSolver(
	AbstractNonlinearProblem* pProblem,
        AbstractNonlinearProblemJacobian* pProblemJacobian,
	const arma::vec* pInitialGuess,
	const ParameterList* pParameterList);

    // Destructor
    ~NewtonSolver();

    // Solution method (implementing pure virtual class)
    void Solve(arma::vec& solution,
	arma::vec& residualHistory,
	ExitFlagType& exitFlag,
  arma::mat* jacobianExeternal=NULL);

    // Accessor for initial guess
    void SetInitialGuess(const arma::vec* pInitialGuess);

    // Accessor for parameter list
    void SetParameterList(const ParameterList* pParameterList);

    // Accessor for nonlinear problem
    void SetProblem(AbstractNonlinearProblem* pProblem);

    // Accessor for jacobian
    void SetProblemJacobian(AbstractNonlinearProblemJacobian* pProblemJacobian);

  private:

    // Hide default constructor
    NewtonSolver();

    // Compute Jacobian via finite differences
    void ComputeDFDU(const arma::vec& u, const arma::vec& f,
	arma::mat& jacobian);

    // Parse parameters and initialise variables before the solution
    void Initialise();

    // Convergence criterion
    ConvergenceCriterion* mpConvergenceCriterion;

    // Initial guess
    const arma::vec* mpInitialGuess;

    // Max number of iterations
    int mMaxIterations;

    // Paramater list
    const ParameterList* mpParameterList;

    // Ouput flag
    bool mPrintOutput;

    // Problem interface
    AbstractNonlinearProblem* mpProblem;
    AbstractNonlinearProblemJacobian* mpProblemJacobian;

    // Tolerance
    double mTolerance;

};


#endif
