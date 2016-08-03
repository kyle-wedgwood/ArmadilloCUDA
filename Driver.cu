#include <iostream>
#include <cstdlib>
#include <cmath>
#include <armadillo>

#include "NewtonSolver.hpp"
#include "Stability.hpp"
#include "EventDrivenMap.hpp"
#include "parameters.hpp"

int main(int argc, char* argv[])
{

  // Parameters
  arma::vec* p_parameters = new arma::vec(1);
  (*p_parameters) << 13.0589f;

  // Instantiate problem
  unsigned int noReal = 1000;
  EventDrivenMap* p_event = new EventDrivenMap(p_parameters,noReal);

  // Initial guess
  arma::vec* p_solution_old = new arma::vec(noSpikes);
  (*p_solution_old) << 0.3310f << 0.6914f << 1.3557f;

  // Newton solver parameter list
  NewtonSolver::ParameterList pars;
  pars.tolerance = 1e-4;
  pars.maxIterations = 10;
  pars.printOutput = true;
  pars.damping = 0.2;

  // Instantiate newton solver (passing Jacobian)
  NewtonSolver* p_newton_solver_1 = new NewtonSolver(p_event, p_solution_old, &pars);

  // Instantiate newton solver (finite differences)
  pars.finiteDifferenceEpsilon = 1e-3;

  // Solve
  arma::vec* p_solution_new = new arma::vec(noSpikes);
  arma::vec* p_residual_history = new arma::vec(); // size assigned by Newton solver
  AbstractNonlinearSolver::ExitFlagType exitFlag;
  std::cout << "\n ******** Solve using finite differences" << std::endl;

  // For computing eigenvalues
  Stability* p_stability = new Stability(Stability::ProblemType::equationFree,p_event);
  arma::mat* p_jacobian = new arma::mat(noSpikes,noSpikes);
  arma::cx_vec* p_eigenvalues = new arma::cx_vec(noSpikes);
  arma::vec* p_real_eigenvalues = new arma::vec(noSpikes);
  int N_steps = 100;
  int numUnstableEigenvalues;

  // For computing Jacobian via finite differences
  arma::vec* p_f0 = new arma::vec(noSpikes);
  arma::vec* p_f1 = new arma::vec(noSpikes);

  /* Now start testing */
  arma::vec* p_test_sol = new arma::vec(noSpikes);
  p_event->ComputeF(*p_solution_old,*p_test_sol);


  /* Do a Newton solve */
  p_newton_solver_1->SetInitialGuess(p_solution_old);
  p_newton_solver_1->Solve(*p_solution_new,*p_residual_history,exitFlag);

  /*
  // Now loop over steps
  for (int i=0;i<N_steps;++i)
  {
    p_newton_solver_1->SetInitialGuess(p_solution_old);
    p_newton_solver_1->Solve(*p_solution_new,*p_residual_history,exitFlag);

    numUnstableEigenvalues = p_stability->ComputeNumUnstableEigenvalues( *p_solution_new);
    std::cout << "Number of unstable eigenvalues = " << numUnstableEigenvalues << std::endl;

    if (numUnstableEigenvalues > 0)
    {
      std::cout << "Solution is unstable" << std::endl;
    }
    else if (numUnstableEigenvalues == 0)
    {
      std::cout << "Solution is stable" << std::endl;

    }

    // Prepare for next step
    (*p_parameters) += 0.1;
    p_event->SetParameters(0,(*p_parameters)[0]);
    *p_solution_old = *p_solution_new;

  }

  // Clean
  */
  delete p_parameters;
  delete p_event;
  /*
  delete p_solution_old;
  delete p_solution_new;
  delete p_residual_history;
  delete p_newton_solver_1;
  delete p_eigenvalues;
  delete p_real_eigenvalues;
  delete p_stability;
  */
}
