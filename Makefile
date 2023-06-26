CXX = c++
TARGET = main

all: $(TARGET) run

$(TARGET): src/$(TARGET).cpp
	$(CXX) -o $(TARGET) src/$(TARGET).cpp

run: $(TARGET)
	./$(TARGET)

clean:
	$(RM) $(TARGET)