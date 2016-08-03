NVCC = nvcc
RM = rm -f
CUFLAGS  = -x cu --std=c++11 -arch=compute_30 -code=sm_30
CPPFLAGS = -x c++ --std=c++11 -arch=compute_30 -code=sm_30
LDFLAGS = -arch=compute_30 -code=sm_30 -lcurand -larmadillo -Xlinker -framework,Accelerate -L/usr/local/lib -L/usr/lib
CUINC  = -I. -dc -I/usr/local/include
INC = -I. -c -I/usr/local/include

OBJS = Driver.o \
       EventDrivenMap.o \
       AbstractNonlinearProblem.o \
       AbstractNonlinearProblemJacobian.o \
       AbstractNonlinearSolver.o \
       NewtonSolver.o \
			 Stability.o \
       ConvergenceCriterion.o \

all: $(OBJS)
		$(NVCC) $(LDFLAGS) $(OBJS) -o Driver

%.o: %.cu
		$(NVCC) $(CUFLAGS) $(CUINC) $< -o $@

%.o: %.cpp
		$(NVCC) $(CPPFLAGS) $(INC) $< -o $@

clean:
		rm -f *.o Driver
