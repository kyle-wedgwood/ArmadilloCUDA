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

  // Perturb solution
  double sigma = 0.01;
  (*p_solution_old) += sigma*randn(noSpikes);

  // Newton solver parameter list
  NewtonSolver::ParameterList pars;
  pars.tolerance = 1e-4;
  pars.maxIterations = 10;
  pars.printOutput = true;
  pars.damping = 0.2;
  pars.finiteDifferenceEpsilon = 1e-3;

  // Instantiate newton solver (finite differences)
  NewtonSolver* p_newton_solver = new NewtonSolver(p_event, p_solution_old, &pars);

  // Solve
  arma::vec* p_solution_new = new arma::vec(noSpikes);
  arma::vec* p_residual_history = new arma::vec(); // size assigned by Newton solver
  AbstractNonlinearSolver::ExitFlagType exitFlag;

  // For computing eigenvalues
  arma::vec noReal(5);
  noReal << 1000 << 500 << 100 << 50 << 10;

  /* Do a Newton solve */
  p_newton_solver->SetInitialGuess(p_solution_old);
  p_newton_solver->Solve(*p_solution_new,*p_residual_history,exitFlag);

  // Save data
  std::ofstream file;
  file.open("ResidualVaryM.dat");

  // Now loop over steps
  for (int i=0;i<noReal.n_elem;++i)
  {
    p_problem->SetNoReal(p_noReal(i));
    p_newton_solver->Solve(*p_solution_new,*p_residual_history,exitFlag);
    for (int j=0;j<(*p_residual_history).n_elem;++j)
    {
      file << (*p_residual_history)(j) << "\t";
    }
    file << "\n";
  }

  // Clean
  file.close();
  delete p_parameters;
  delete p_event;
  delete p_solution_old;
  delete p_solution_new;
  delete p_residual_history;
  delete p_newton_solver;
}

