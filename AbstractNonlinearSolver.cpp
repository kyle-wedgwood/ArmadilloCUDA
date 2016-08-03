#include <cassert>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <sstream>


#include "AbstractNonlinearSolver.hpp"

// Printing infos when iterations begin
void AbstractNonlinearSolver::
PrintHeader(const std::string solverName,
    int maxIterations, double tolerance) const
{
  std::cout << "------------------------------------------------"
            << std::endl
            << " Attempt to solve nonlinear problem with " + solverName
            << std::endl
	    << " max number of iterations = " << maxIterations
            << std::endl
	    << " tolerance = " << tolerance
            << std::endl
            << "------------------------------------------------"
            << std::endl;
}

// Printing infos when iterations terminate
void AbstractNonlinearSolver::
PrintFooter(const int iteration, const ExitFlagType exitFlag) const
{
  
  switch (exitFlag)
  {
    case ExitFlagType::converged :
    std::cout << "------------------------------------------------"
              << std::endl
              << "The method converged after "  
              << iteration
              << " iterations"
              << std::endl;
    break;
    case ExitFlagType::notConverged:
    std::cout << "------------------------------------------------"
              << std::endl
              << "The method failed to converge after "  
              << iteration
              << " iterations"
              << std::endl;
    break;
    default:
    std::cout << "Exit flag not known" << std::endl;
  }
//   if (converged)
//   {
//     std::cout << "------------------------------------------------"
// 	      << std::endl
//               << "The method converged after "  
//               << iteration
// 	      << " iterations"
// 	      << std::endl;
//   }
//   else
//   {
//     std::cout << "------------------------------------------------"
// 	      << std::endl
//               << "The method failed to converge after "  
//               << iteration
// 	      << " iterations"
// 	      << std::endl;
//   }

}

// Printing infos about current iteration
void AbstractNonlinearSolver::
PrintIteration(const int iteration, const double errorEstimate,
    const bool initialise) const
{

  // At the first iterate, print the table header
  if (initialise)
  {
    std::cout << std::setw(10) << "Iteration"
              << std::setw(25) << "error estimate"
              << std::endl;
  }

  // Residual norm is printed with scientific notation
  std::cout << std::setw(10) << iteration
            << std::scientific
            << std::setprecision(6) 
            << std::setw(25) << errorEstimate
            << std::endl;

}
