CXX = g++
CXXFLAGS = -std=c++17 -Wall -O2
SRC = src/main.cpp
TARGET = hello

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SRC)

clean:
	rm -f $(TARGET)
