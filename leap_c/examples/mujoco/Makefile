# Makefile for MuJoCo-Acados integration example
# This builds a standalone test executable

# Project root (relative to this Makefile)
PROJECT_ROOT = ../../..

# Paths
ACADOS_INCLUDE = $(PROJECT_ROOT)/external/acados/include
ACADOS_LIB = $(PROJECT_ROOT)/external/acados/lib
MUJOCO_INCLUDE = /opt/homebrew/envs/leapc_gemini/lib/python3.11/site-packages/mujoco/include
MUJOCO_LIB = /opt/homebrew/envs/leapc_gemini/lib/python3.11/site-packages/mujoco

# Compiler settings
CC = gcc
CFLAGS = -std=c99 -O2 -Wall -fPIC
CPPFLAGS = -I$(ACADOS_INCLUDE) \
           -I$(ACADOS_INCLUDE)/acados \
           -I$(ACADOS_INCLUDE)/blasfeo/include \
           -I$(ACADOS_INCLUDE)/hpipm/include \
           -I$(MUJOCO_INCLUDE)

LDFLAGS = -L$(ACADOS_LIB) -L$(MUJOCO_LIB)
LDLIBS = -lacados -lhpipm -lblasfeo -lm
MUJOCO_DYLIB = $(MUJOCO_LIB)/libmujoco.3.3.2.dylib

# For macOS: add rpath so libraries can be found at runtime
ifeq ($(shell uname -s),Darwin)
    LDFLAGS += -Wl,-rpath,$(ACADOS_LIB) -Wl,-rpath,$(MUJOCO_LIB)
endif

# Source files
SRC = leapc_mujoco.c test_mujoco_dyn.c
OBJ = $(SRC:.c=.o)
TEST_EXE = test_mujoco_dyn

# Targets
.PHONY: all clean test

all: $(TEST_EXE)

$(TEST_EXE): $(OBJ)
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS) $(LDLIBS) $(MUJOCO_DYLIB)

%.o: %.c
	$(CC) $(CFLAGS) $(CPPFLAGS) -c $< -o $@

# Helper target to run test with a model file
# Usage: make test MODEL=path/to/model.xml
test: $(TEST_EXE)
	@if [ -z "$(MODEL)" ]; then \
		echo "Error: Please specify MODEL=path/to/model.xml"; \
		echo "Example: make test MODEL=cartpole.xml"; \
		exit 1; \
	fi
	./$(TEST_EXE) $(MODEL)

clean:
	rm -f $(OBJ) $(TEST_EXE)

# Display help
help:
	@echo "MuJoCo-Acados Integration Example"
	@echo ""
	@echo "Targets:"
	@echo "  all    - Build the test executable"
	@echo "  test   - Run the test (requires MODEL=path/to/model.xml)"
	@echo "  clean  - Remove build artifacts"
	@echo "  help   - Show this help message"
	@echo ""
	@echo "Example usage:"
	@echo "  make"
	@echo "  make test MODEL=test_model.xml"
	@echo "  ./run_test.sh test_model.xml"
