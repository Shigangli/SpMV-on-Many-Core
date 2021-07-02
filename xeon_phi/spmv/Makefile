INCDIR	     ?= .
# Includes
INCLUDES  += -I$(INCDIR) -I/usr/include/CL
OCLLIBDIR := -lOpenCL 

LINK = icc  -g -O3 

TARGET = test
SRCS = ../init.cpp test.cpp 
$(TARGET):
	$(LINK) -o $@ $(INCLUDES) $(SRCS) $(OCLLIBDIR) -O3 -D DEBUG 
clean:
	rm -f $(TARGET) 

