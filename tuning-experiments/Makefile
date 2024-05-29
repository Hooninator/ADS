#Compiler settings
CXX = mpicxx 
CXXFLAGS =  -w -O3 -fopenmp

# Target executable
TARGET = combblas-spgemm

# Source files
SOURCES = CombBLASSpGEMM.cpp

# Object files
OBJECTS = $(SOURCES:.cpp=.o)

INCLUDE = -I$(HOME)/CombBLAS/install/include  \
		  -I/usr/local/cuda/include


LINK = -L$(HOME)/CombBLAS/install/lib -lCombBLAS -lGraphGenlib -lUsortlib\

# Default target
all: $(TARGET)

# Linking
$(TARGET): $(OBJECTS)
	$(CXX) $(CXXFLAGS) -o $(TARGET)  $(OBJECTS) $(LINK)
# Compiling
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< $(INCLUDE)

# Clean up
clean:
	rm -f $(OBJECTS) $(TARGET)
