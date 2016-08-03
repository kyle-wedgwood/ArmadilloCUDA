#include <iostream>
#include <cmath>
#include <armadillo>

#include "NewtonSolver.hpp"
#include "Fold.hpp"

int main(int argc, char* argv[])
{

  // Parameters
  arma::vec* p_parameters = new arma::vec(1);
  (*p_parameters) << 2.5;

  // // Instantiate problem
  Fold* p_fold = new Fold(p_parameters);

  // Initial guess
  arma::vec* p_initial_guess = new arma::vec(2);
  (*p_initial_guess) << 1.0 << 1.0;

  // Newton solver parameter list
  NewtonSolver::ParameterList pars;
  pars.tolerance = 1e-10;
  pars.maxIterations = 10;
  pars.printOutput = true;

  // Instantiate newton solver (passing Jacobian)
  NewtonSolver* p_newton_solver_1 = new NewtonSolver(p_fold, p_fold, p_initial_guess, &pars);

  // Solve
  arma::vec* p_solution = new arma::vec(2);
  arma::vec* p_residual_history = new arma::vec(); // size assigned by Newton solver
  AbstractNonlinearSolver::ExitFlagType exitFlag;
  std::cout << "\n ******** Solve using Jacobian" << std::endl;
  p_newton_solver_1->Solve(*p_solution,*p_residual_history,exitFlag);
  std::cout << "\n solution = \n" << *p_solution;

  // Instantiate newton solver (Jacobian computed by finite differences)
  pars.finiteDifferenceEpsilon = 1e-1;
  NewtonSolver* p_newton_solver_2 = new NewtonSolver(p_fold, p_initial_guess, &pars);

  // Solve using finite differences jacobian
  std::cout << "\n ******** Solve using finite difference" << std::endl;
  p_newton_solver_2->Solve(*p_solution,*p_residual_history,exitFlag);
  std::cout << "\n solution = \n" << *p_solution;

  // Change initial guess and solve again (use the same solver)
  std::cout << "\n ******** Solve using finite difference from new initial guess" << std::endl;
  (*p_initial_guess) << 0.1 << 0.1;
  p_newton_solver_2->SetInitialGuess(p_initial_guess);
  p_newton_solver_2->Solve(*p_solution,*p_residual_history,exitFlag);
  std::cout << "\n solution = \n" << *p_solution;

  // Change number of iterations guess and solve again (use the same solver, which fails)
  std::cout << "\n ******** Solve using only a few iterations" << std::endl;
  pars.maxIterations = 5;
  p_newton_solver_2->SetParameterList(&pars);
  p_newton_solver_2->Solve(*p_solution,*p_residual_history,exitFlag);
  std::cout << "\n solution = \n" << *p_solution;

  // Try again, loosen tolerance and increase number of iterates (converges)
  std::cout << "\n ******** Solve using a coarse tolerance" << std::endl;
  pars.tolerance = 1e-1;
  p_newton_solver_2->SetParameterList(&pars);
  p_newton_solver_2->Solve(*p_solution,*p_residual_history,exitFlag);
  std::cout << "\n solution = \n" << *p_solution;

  // Switch the second solver to use a user-supplied jacobian
  pars.tolerance = 1e-10;
  pars.maxIterations = 10;
  p_newton_solver_2->SetProblemJacobian(p_fold);
  p_newton_solver_2->Solve(*p_solution,*p_residual_history,exitFlag);
  std::cout << "\n solution = \n" << *p_solution;

  // Clean
  delete p_parameters;
  delete p_fold;
  delete p_initial_guess;
  delete p_solution;
  delete p_residual_history;
  delete p_newton_solver_1;
  delete p_newton_solver_2;

}
