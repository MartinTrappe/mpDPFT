NUM_CORES := $(shell nproc)

CXX = g++
CXXSTD = -std=c++20 -static
CXXFLAGS = -I/opt/etsf/include -I${CURDIR}/Eigen_Headers -I${CURDIR}/utils -fopenmp -D_GLIBCXX_PARALLEL -mtune=native -march=native -funroll-loops -fconcepts -Wno-volatile
#for NSCC aspire2
#CXXFLAGS = -I/home/users/nus/cqtmst/scratch/gsl/include -I/home/users/nus/cqtmst/scratch/libxc/include -I/home/users/nus/cqtmst/scratch/fftw/include -I/opt/etsf/include -I${CURDIR}/Eigen_Headers -fopenmp -D_GLIBCXX_PARALLEL -mtune=native -march=native -funroll-loops -fconcepts -Wno-volatile
CXXOPTFLAGS = -flto=$(shell nproc) -O3

# Check if g++ with -std=c++20 support is available
GPP_VERSION := $(shell g++ --version | awk '/g\+\+/ {print $$NF}' | cut -d. -f1)
MIN_GPP_VERSION = 10

ifeq ($(shell test $(GPP_VERSION) -ge $(MIN_GPP_VERSION) && echo "yes"),yes)
    $(info g++ with -std=c++20 support is available (version $(GPP_VERSION)))
else
    $(info g++ with -std=c++20 support is not available or version is too low (version $(GPP_VERSION)))
    $(info Switching to an older standard (-std=c++17))
    CXXSTD = -std=c++17
endif

#################### BEGIN USER INPUT ####################

#default:
CC = $(CXX) $(CXXSTD) $(CXXFLAGS) $(CXXOPTFLAGS)
#for gdb & valgrind:
#CC = $(CXX) -ggdb3 $(CXXSTD) $(CXXFLAGS) -O2 -fno-inline -fno-omit-frame-pointer
#gdb with sanitation:
#CC = $(CXX) -fsanitize=address -fsanitize=undefined -ggdb3 $(CXXSTD) $(CXXFLAGS) -O0 -fno-inline -fno-omit-frame-pointer
#with production code optimization:
#CC = $(CXX) -ggdb3 $(CXXSTD) $(CXXFLAGS) $(CXXOPTFLAGS)
#for using cSA:
#CC = $(CXX) $(CXXSTD) $(CXXFLAGS) $(CXXOPTFLAGS) -pthread

#default:
LDFLAGS= -L${EBROOTGSL} -L/usr -L/usr/local/lib -L/opt/etsf -L/opt/etsf/lib -L/usr/include -lgsl -lgslcblas -lmpfr -lgmp -lfftw3_omp -lfftw3 -lm -lxc -funroll-loops
#for NSCC aspire2
#LDFLAGS= -L${EBROOTGSL} -L/usr -L/home/users/nus/cqtmst/scratch/gsl/lib -L/home/users/nus/cqtmst/scratch/libxc/lib -L/home/users/nus/cqtmst/scratch/fftw/lib -L/opt/etsf -L/opt/etsf/lib -L/usr/include -lgsl -lgslcblas -lmpfr -lgmp -lfftw3_omp -lfftw3 -lm -lxc -funroll-loops
#for using cSA:
#LDFLAGS= -L${EBROOTGSL} -L/usr -L/usr/local/lib -L/opt/etsf -L/opt/etsf/lib -L/usr/include -lgsl -lgslcblas -lmpfr -lgmp -lfftw3_omp -lfftw3 -lm -lxc -funroll-loops -pthread

#################### END USER INPUT ####################

OBJS = mpDPFTmain.o mpDPFT.o Plugin_KD.o Plugin_Triangulation.o Plugin_1pEx.o Plugin_1pEx_Rho1p.o Plugin_OPT.o Plugin_cec14_test_func.o statistics.o specialfunctions.o solvers.o optimization.o linalg.o kernels_sse2.o kernels_fma.o kernels_avx2.o interpolation.o integration.o fasttransforms.o diffequations.o dataanalysis.o ap.o alglibmisc.o alglibinternal.o utils/interactions.o
HEADERS = stdafx.h statistics.h specialfunctions.h solvers.h optimization.h linalg.h kernels_sse2.h kernels_fma.h kernels_avx2.h interpolation.h integration.h fasttransforms.h diffequations.h dataanalysis.h ap.h alglibmisc.h alglibinternal.h utils/interactions.h

.PHONY: all clean

all: mpDPFT_michael

mpDPFT_michael: $(OBJS)
	@echo "Linking mpDPFT..."
	@time $(CC) $(OBJS) -o $@ $(LDFLAGS)

# %.o: %.cpp $(HEADERS)
# 	$(CC) -c $< -o $@
# Auto-generate dependencies
DEPFILES = $(OBJS:.o=.d)

-include $(DEPFILES)

%.o: %.cpp
	$(CC) -MMD -MP -c $< -o $@

TEST_SRC = tests/Plugin_KD_test.cpp
TEST_OBJ = $(TEST_SRC:.cpp=.o)
TEST_BIN = Plugin_KD_test
TEST_INCLUDES = -I.. -I.

# Build test executable
$(TEST_BIN): $(TEST_OBJ) Plugin_KD.o
	@echo "Linking test executable..."
	$(CC) $(TEST_OBJ) Plugin_KD.o -o $(TEST_BIN) $(LDFLAGS)

# Compile test object
$(TEST_OBJ): $(TEST_SRC)
	@echo "Compiling test: $<"
	$(CC) -MMD -MP $(TEST_INCLUDES) -c $< -o $@

# Run the test
test: $(TEST_BIN)
	@echo "Running Plugin_KD_test..."
	@./$(TEST_BIN)

.PHONY: clean
clean:
	rm -f mpDPFT_michael \
	       $(OBJS) \
	       $(OBJS:.o=.d)



# DIFF_SRC = tests/differential_equation.cpp
# DIFF_OBJ = $(DIFF_SRC:.cpp=.o)
# ALGLIB_SRCS = ap.cpp alglibmisc.cpp optimization.cpp interpolation.cpp
# PLUGIN_SRC = Plugin_KD.cpp
# DIFF_BIN = differential_equation
# DIFF_INCLUDES = -I. -I${CURDIR}/Eigen_Headers -I/opt/etsf/include -I${CURDIR}

# # Build differential executable
# $(DIFF_BIN): $(DIFF_OBJ)
# 	@echo "Linking differential..."
# 	$(CC) $(DIFF_OBJ) $(ALGLIB_SRCS) $(PLUGIN_SRC) -o $(DIFF_BIN) $(LDFLAGS)


# # Compile differential object
# $(DIFF_OBJ): $(DIFF_SRC)
# 	@echo "Compiling differential: $<"
# 	$(CC) -MMD -MP $(DIFF_INCLUDES) -c $< -o $@

# # Run differential
# run_diff: $(DIFF_BIN)
# 	@echo "Running differential..."
# 	@./$(DIFF_BIN)

#################### Differential Equation Build ####################

############################
# Sources
############################

DIFF_SRC      = tests/differential_equation.cpp
PLUGIN_SRC    = Plugin_KD.cpp

ALGLIB_SRCS   := $(wildcard alglib/src/*.cpp)

############################
# Objects
############################

DIFF_OBJ      = $(DIFF_SRC:.cpp=.o)
PLUGIN_OBJ    = $(PLUGIN_SRC:.cpp=.o)
ALGLIB_OBJS   = $(ALGLIB_SRCS:.cpp=.o)

############################
# Executable
############################

DIFF_BIN = differential_equation

############################
# Includes
############################

DIFF_INCLUDES = -I. \
                -I${CURDIR}/Eigen_Headers \
                -I/opt/etsf/include \
                -I${CURDIR}/alglib/src \

############################
# Build executable
############################

$(DIFF_BIN): $(DIFF_OBJ) $(PLUGIN_OBJ) $(ALGLIB_OBJS)
	@echo "Linking differential..."
	$(CXX) $^ -o $@ $(LDFLAGS) -fopenmp

############################
# Compile rules
############################

%.o: %.cpp
	@echo "Compiling $<"
	$(CXX) -MMD -MP $(DIFF_INCLUDES) $(TEST_INCLUDES) -c $< -o $@

############################
# Run
############################

run_diff: $(DIFF_BIN)
	./$(DIFF_BIN)

############################
# Clean
############################

.PHONY: clean
clean:
	rm -f $(DIFF_OBJ) $(PLUGIN_OBJ) $(ALGLIB_OBJS) \
	      $(DIFF_BIN) \
	      $(DIFF_OBJ:.o=.d) \
	      $(PLUGIN_OBJ:.o=.d) \
	      $(ALGLIB_OBJS:.o=.d)