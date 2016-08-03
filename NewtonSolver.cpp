#include "NewtonSolver.hpp"
#include <cassert>
#include <cmath>
#include <iostream>

// Constructor (Jacobian computed by finite differences explicitly)
NewtonSolver::NewtonSolver(
    AbstractNonlinearProblem* pProblem,
    const arma::vec* pInitialGuess,
    const ParameterList* pParameterList)
{
  mpProblem = pProblem;
  mpProblemJacobian = NULL;
  mpInitialGuess = pInitialGuess;
  mpParameterList = pParameterList;
  mpConvergenceCriterion = NULL;
}

// Constructor (Jacobian passed by user)
NewtonSolver::NewtonSolver(
    AbstractNonlinearProblem* pProblem,
    AbstractNonlinearProblemJacobian* pProblemJacobian,
    const arma::vec* pInitialGuess,
    const ParameterList* pParameterList)
{
  mpProblem = pProblem;
  mpProblemJacobian = pProblemJacobian;
  mpInitialGuess = pInitialGuess;
  mpParameterList = pParameterList;
  mpConvergenceCriterion = NULL;
}

// Destructor
NewtonSolver::~NewtonSolver()
{
  delete mpConvergenceCriterion;
}

// Solution method
void NewtonSolver::Solve(arma::vec& solution,
    arma::vec& residualHistory,
    ExitFlagType& exitFlag,
    arma::mat* pJacobianExternal)
{

  // Parse parameter list
  Initialise();

  // Eventually print the header
  if (mPrintOutput)
  {
    PrintHeader("Newton Method", mMaxIterations, mTolerance);
  }

  // Check if dimensions are compatible
  int problem_size = mpInitialGuess->n_rows;
  assert( problem_size == solution.n_rows );

  // Iterator
  int iteration = 0;

  // Assign initial guess
  solution = *mpInitialGuess;

  // Initial residual
  arma::vec residual(problem_size);
  mpProblem->ComputeF(solution,residual);

  // Residual norm
  double residual_norm = arma::norm(residual,2);

  // Residual history
  residualHistory.set_size(1+mpParameterList->maxIterations);
  residualHistory(iteration) = residual_norm;

  // Eventually print iteration
  if (mPrintOutput)
  {
    PrintIteration(iteration, residual_norm, true);
  }

  // Check convergence
  bool converged = mpConvergenceCriterion->TestConvergence(residual_norm);

  arma::mat jacobian(problem_size,problem_size);
  // Newton's method main loop
  while ( (iteration < mMaxIterations) && (!converged) )
  {

    // Compute Jacobian (eventually use finite differences)
    if (mpProblemJacobian)
    {
      mpProblemJacobian->ComputeDFDU(solution,jacobian);
    }
    else
    {
      ComputeDFDU(solution,residual,jacobian);
    }

    // Linear solve to find direction
    arma::vec direction = arma::solve( jacobian, -residual);

    // Update solution
    solution += mpParameterList->damping*direction;

    // Update iterator
    iteration++;

    // Update residual
    mpProblem->ComputeF(solution,residual);

    // Update residual norm
    residual_norm = arma::norm(residual,2);

    // Check convergence
    converged = mpConvergenceCriterion->TestConvergence(residual_norm);

    // Update residual history
    residualHistory(iteration) = residual_norm;

    // Print infos
    if (mPrintOutput)
    {
      PrintIteration(iteration, residual_norm);
    }


  }

  // Trim residual history
  residualHistory.head(iteration);

  // Assign exit flag
  if (converged)
  {
    exitFlag = AbstractNonlinearSolver::ExitFlagType::converged;
  }
  else
  {
    exitFlag = AbstractNonlinearSolver::ExitFlagType::notConverged;
  }

  // Eventually print final message
  if (mPrintOutput)
  {
    PrintFooter(iteration, exitFlag);
  }

  // Output to external jacobian if requested
  if (pJacobianExternal)
  {
    assert( (*pJacobianExternal).n_rows==problem_size);
    assert( (*pJacobianExternal).n_cols==problem_size);
    assert( jacobian.n_rows==problem_size);
    (*pJacobianExternal) = jacobian;
  }

}

// Compute Jacobian via finite differences
void NewtonSolver::
ComputeDFDU(const arma::vec& u, const arma::vec& f, arma::mat& jacobian)
{

  // Perturbed solution
  arma::vec du(u);

  // Problem size
  int problem_size = mpInitialGuess->n_rows;

  // Perturbed residual
  arma::vec df(problem_size);

  // Epsilon
  double epsilon = mpParameterList->finiteDifferenceEpsilon;

  // For each column
  for (int i=0; i<problem_size; i++)
  {
    // Restore original solution, then perturb it
    if (i>0)
    {
      du(i-1) = u(i-1);
    }
    du(i) += epsilon;

    // Perturbed residual
    mpProblem->ComputeF(du,df);

    // Assign jacobian column
    jacobian.col(i) = (df - f) * pow(epsilon,-1);
  }

}

// Accessor for initial guess
void NewtonSolver::SetInitialGuess(const arma::vec* pInitialGuess)
{
  mpInitialGuess = pInitialGuess;
}

// Accessor for parameters
void NewtonSolver::SetParameterList(const ParameterList* pParameterList)
{
  mpParameterList = pParameterList;
}

void NewtonSolver::SetProblem(AbstractNonlinearProblem* pProblem)
{
  mpProblem = pProblem;
}

void NewtonSolver::SetProblemJacobian(AbstractNonlinearProblemJacobian* pProblemJacobian)
{
  mpProblemJacobian = pProblemJacobian;
}

// Parse parameter list and initialise other parameters before solution
void NewtonSolver::Initialise()
{

  // Parse parameter list
  mMaxIterations = mpParameterList->maxIterations;
  mPrintOutput = mpParameterList->printOutput;
  mTolerance = mpParameterList->tolerance;

  // Convergence critierion
  if (mpConvergenceCriterion)
  {
    mpConvergenceCriterion->SetTolerance(mTolerance);
  }
  else
  {
    mpConvergenceCriterion = new ConvergenceCriterion(mTolerance);
  }

}

