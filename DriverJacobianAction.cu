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
  EventDrivenMap* p_problem = new EventDrivenMap(p_parameters,noReal);

  // Initial guess
  arma::vec u0 = arma::vec(noSpikes);
  arma::vec u1 = arma::vec(noSpikes);
  u0 << 0.3310f << 0.6914f << 1.3557f;

  // For computing Jacobian via finite differences
  arma::vec f0 = arma::vec(noSpikes);
  arma::vec f1 = arma::vec(noSpikes);
  arma::mat jac = arma::mat(noSpikes,noSpikes);

  // Vector of epsilons
  unsigned int N_steps = 100;
  arma::vec epsilon  = arma::vec(N_steps,fill::zeros);

  double eps_max = log(1.0e-1);
  double eps_min = log(1.0e-5);
  double deps = (eps_max-eps_min)/(N_steps-1);
  double epsilon = eps_min;
  double matrix_action_norm;

  // Test vector
  arma::vec test_vec = arma::vec(noSpikes,fill::randn);

  // File to save data
  std::ofstream file;
  file.open("MatrixAction.dat");
  file << "EPS" << "JV" << "\r\n";

  // Now loop over steps
  for (int i=0;i<N_steps;++i)
  {
    epsilon += deps;
    epsilon = pow(10,eps);
    p_problem->ComputeF(u0,f0);

    // Construct Jacobian
    for (int j=0;j<noSpikes;++j)
    {
      if (j>0)
      {
        u1(j-1) = u0(j-1);
      }
      u1 += epsilon;
      p_problem->ComputeF(u1,f1);

      jac.col(i) = (f1-f0)*pow(epsilon,-1)
    }

    // Calculate Jacobian action
    matrix_action_norm = norm(jac*test_vec,2);

    // Save and display data
    file << epsilon << matrix_action_norm << "\r\n";
    std::cout << epsilon << matrix_action_norm << std::endl;

    // Prepare for next step
    epsilon = log10(epsilon);
  }

  // Clean
  file.close();
  delete p_parameters;
  delete p_problem;
}
