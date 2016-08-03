#include "Stability.hpp"
#include <armadillo>
#include "AbstractNonlinearProblem.hpp"
#include "AbstractNonlinearProblemJacobian.hpp"

Stability::Stability( ProblemType type, AbstractNonlinearProblem *pProblem)
{
  mpProblem = pProblem;
  mpProblemJacobian = NULL;
  mProblemType = type;
}

Stability::Stability( ProblemType type,
                      AbstractNonlinearProblem *pProblem,
                      AbstractNonlinearProblemJacobian *pProblemJacobian)
{
  mpProblem         = pProblem;
  mpProblemJacobian = pProblemJacobian;
  mProblemType = type;
}

int Stability::ComputeNumUnstableEigenvalues(const arma::vec& u)
{
  // Find eigenvalues
  arma::cx_vec eigenvalues = ComputeEigenvalues(u);

  if (mProblemType == ProblemType::flow)
  {
    return accu(arma::real(eigenvalues)>0.0);
  }
  else
  {
    return accu(abs(eigenvalues)>1.0);
  }
}

int Stability::ComputeNumUnstableEigenvalues( const arma::mat& jacobian)
{
  // Find eigenvalues
  arma::cx_vec eigenvalues = eig_gen(jacobian);

  if (mProblemType == ProblemType::flow)
  {
    return accu(arma::real(eigenvalues)>0.0);
  }
  else
  {
    return accu(abs(eigenvalues)>1.0);
  }
}

arma::cx_vec Stability::ComputeEigenvalues(const arma::vec& u)
{
  // Problem size
  int problem_size = u.n_rows;

  arma::mat jacobian(problem_size,problem_size);

  if (mpProblemJacobian)
  {
    mpProblemJacobian->ComputeDFDU(u,jacobian);
  }
  else
  {
    ComputeDFDU(u,jacobian);
  }

  if (mProblemType == ProblemType::equationFree)
  {
    jacobian += arma::mat(problem_size,problem_size,arma::fill::eye);
  }
  return eig_gen(jacobian);

}

void Stability::ComputeDFDU(const arma::vec& u, arma::mat& jacobian)
{

  // Perturbed solution
  arma::vec du(u);

  // Problem size
  int problem_size = u.n_rows;

  // Perturbed residual
  arma::vec f(problem_size);
  arma::vec df(problem_size);

  // Epsilon
  double epsilon = mFiniteDifferenceEpsilon;

  mpProblem->ComputeF(u,f);

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
