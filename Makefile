# Compiler and Flags
CXX := g++
NVCC := nvcc
CXXFLAGS := -std=c++11 -Wall -O2 -fno-inline
LDFLAGS := -L/usr/local/cuda-12.5/lib64 -lcudart -ldl -lcuda
DEBUGFLAGS := -g -O0

# OpenCV Configuration
OPENCV_CFLAGS := $(shell pkg-config --cflags opencv4)
OPENCV_LIBS := $(shell pkg-config --libs opencv4)

# Paths
INCLUDE_DIRS := -I/usr/local/cuda-12.5/include $(OPENCV_CFLAGS)

# Files
SRC := main.cpp flowvec.cpp 
OBJS := $(patsubst %.cpp, %.o, $(SRC)) $(CUSRC:.cu=.o)
TARGET := ofvec
SHARED_LIB := libflowvec.so

# Rules
.PHONY: all clean

all: $(TARGET)

# Build the main executable
$(TARGET): $(OBJS)
	$(CXX) $(DEBUGFLAGS) $(CXXFLAGS) -o $@ $^ $(INCLUDE_DIRS) $(LDFLAGS) $(OPENCV_LIBS)

# Build shared library
$(SHARED_LIB): flowvec.o kernel.o
	$(CXX) $(DEBUGFLAGS) -shared -o $@ $^ $(LDFLAGS)

# Compile source files
%.o: %.cpp
	$(CXX) $(DEBUGFLAGS) $(CXXFLAGS) -c $< -o $@ $(INCLUDE_DIRS)

# Clean up
clean:
	rm -rf $(TARGET) $(SHARED_LIB) *.o
